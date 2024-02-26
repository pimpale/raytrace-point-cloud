vulkano_shaders::shader! {
    linalg_type: "nalgebra",
    ty: "compute",
    src: r"
        #version 460
        #extension GL_EXT_ray_query: require
        #extension GL_EXT_scalar_block_layout: require
        #extension GL_EXT_buffer_reference2: require
        #extension GL_EXT_shader_explicit_arithmetic_types_int64: require
        #extension GL_EXT_shader_explicit_arithmetic_types_int8: require
        #extension GL_EXT_nonuniform_qualifier: require

        #define M_PI 3.1415926535897932384626433832795

        layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

        layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;
        
        layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer Vertex {
            vec3 min;
            vec3 max;
        };

        layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer GaussianSplat {
            vec4 rot;
            vec3 scale;
            vec3 color;
            float opacity;
        };

        struct InstanceData {
            // points to the device address of the vertex data for this instance
            uint64_t vertex_buffer_addr;
            // points to the device address of the gaussian splat data for this instance
            uint64_t gaussian_splat_buffer_addr;
            // the transform of this instance
            mat4x3 transform;
        };

        layout(set = 0, binding = 1, scalar) readonly buffer InstanceDataBuffer {
            InstanceData instance_data[];
        };

        layout(set = 0, binding = 2, scalar) writeonly buffer Outputs {
            u8vec4 out_color[];
        };

        struct Camera {
            vec3 eye;
            vec3 front;
            vec3 up;
            vec3 right;
            uvec2 screen_size;
        };

        layout(push_constant, scalar) uniform PushConstants {
            Camera camera;
            uint frame;
        } push_constants;


        // source: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
        // Construct a float with half-open range [0:1] using low 23 bits.
        // All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
        float floatConstruct( uint m ) {
            const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
            const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32
        
            m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
            m |= ieeeOne;                          // Add fractional part to 1.0
        
            float  f = uintBitsToFloat( m );       // Range [1:2]
            return f - 1.0;                        // Range [0:1]
        }

        // accepts a seed, h, and a 32 bit integer, k, and returns a 32 bit integer
        // corresponds to the loop in the murmur3 hash algorithm
        // the output should be passed to murmur3_finalize before being used
        uint murmur3_combine(uint h, uint k) {
            // murmur3_32_scramble
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;

            h ^= k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xe6546b64;
            return h;
        }

        // accepts a seed, h and returns a random 32 bit integer
        // corresponds to the last part of the murmur3 hash algorithm
        uint murmur3_finalize(uint h) {
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            return h;
        }

        uint murmur3_combinef(uint h, float k) {
            return murmur3_combine(h, floatBitsToUint(k));
        }

        float murmur3_finalizef(uint h) {
            return floatConstruct(murmur3_finalize(h));
        }

        // https://gist.github.com/pezcode/150eb97dd41b67b611d0de7bae273e98
        mat3 quat_to_mat3(vec4 q) {
            // multiply by sqrt(2) to get rid of all the 2.0 factors in the matrix
            q *= 1.414214;
        
            float xx = q.x*q.x;
            float xy = q.x*q.y;
            float xz = q.x*q.z;
            float xw = q.x*q.w;
        
            float yy = q.y*q.y;
            float yz = q.y*q.z;
            float yw = q.y*q.w;
        
            float zz = q.z*q.z;
            float zw = q.z*q.w;
        
            return mat3(
                1.0 - yy - zz,
                xy + zw,
                xz - yw,
                
                xy - zw,
                1.0 - xx - zz,
                yz + xw,
        
                xz + yw,
                yz - xw,
                1.0 - xx - yy);
        }

        // inverse of the cumulative distribution function of the normal distribution
        float invcdf(float p) {
            if(p > 0.5) {
                return 5.55556 * (1 - pow((1-p)/p, 0.1186));
            } else {
                return -5.55556 * (1 - pow(p/(1-p), 0.1186));
            }
        }

        // returns a vector sampled from the hemisphere with positive y
        // sample is weighted by cosine of angle between sample and y axis
        // https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_08_PathTracing.pdf
        vec3 cosineWeightedSampleHemisphere(vec2 uv) {
            float z = uv.x;
            float r = sqrt(max(0, 1.0 - z));
            float phi = 2.0 * M_PI * uv.y;
            
            return vec3(r * cos(phi), sqrt(z), r * sin(phi));
        }

        struct IntersectionCoordinateSystem {
            vec3 normal;
            vec3 tangent;
            vec3 bitangent;
        };


        // returns a vector sampled from the hemisphere defined around the coordinate system defined by normal, tangent, and bitangent
        // normal, tangent and bitangent form a right handed coordinate system 
        vec3 alignedCosineWeightedSampleHemisphere(vec2 uv, IntersectionCoordinateSystem ics) {
            vec3 hemsam = cosineWeightedSampleHemisphere(uv);
            return normalize(hemsam.x * ics.tangent + hemsam.y * ics.normal + hemsam.z * ics.bitangent);
        }

        struct IntersectionInfo {
            bool miss;
            IntersectionCoordinateSystem hit_coords;
            float tmin;
            float tmax;
            uint instance_index;
            uint primitive_index;
        };

        IntersectionInfo getIntersectionInfo(vec3 origin, vec3 direction, uint ignore_instance, uint ignore_primitive) {
            uvec2 ignore_id = uvec2(ignore_instance, ignore_primitive);
            const float t_min = 0.05;
            const float t_max = 1000.0;
            rayQueryEXT ray_query;
            rayQueryInitializeEXT(
                ray_query,
                top_level_acceleration_structure,
                gl_RayFlagsNoneEXT,
                0xFF,
                origin,
                t_min,
                direction,
                t_max
            );

            // we have to pack all of our data to work around a bizzare bug in Nvidia's ray tracing implementation
            // basically, we suffer from a  CTX SWITCH TIMEOUT if we assign more than one variable in the while loop
            // also, the ray query does not pick the closest intersection, so we have to do that manually
            mat4x3 packed_data = mat4x3(
                vec3(
                    // minimum t
                    t_max,
                    // maximum t
                    t_max,
                    // dummy
                    0.0
                ),
                vec3(
                    // instance index
                    0.0,
                    // primitive index
                    0.0,
                    // dummy
                    0.0
                ),
                // normal
                vec3(0.0),
                // tangent
                vec3(0.0)
            );

            // trace ray
            while (rayQueryProceedEXT(ray_query)) {
                // commit if intersection with aabb
                if(rayQueryGetIntersectionCandidateAABBOpaqueEXT(ray_query)) {
                    vec3 candidate_object_space_origin = rayQueryGetIntersectionObjectRayOriginEXT(ray_query, false);
                    vec3 candidate_object_space_direction = rayQueryGetIntersectionObjectRayDirectionEXT(ray_query, false);

                    // get the primitive index and instance index
                    uint candidate_prim_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, false);
                    uint candidate_instance_index = rayQueryGetIntersectionInstanceIdEXT(ray_query, false);
                    uvec2 candidate_id = uvec2(candidate_instance_index, candidate_prim_index);
    
                    Vertex candidate_v = Vertex(instance_data[candidate_instance_index].vertex_buffer_addr)[candidate_prim_index];

                    vec3 inverse_dir = 1.0 / candidate_object_space_direction;
                    vec3 tbot = inverse_dir * (candidate_v.min - candidate_object_space_origin);
                    vec3 ttop = inverse_dir * (candidate_v.max - candidate_object_space_origin);
                    vec3 tmin = min(ttop, tbot);
                    vec3 tmax = max(ttop, tbot);
                    vec2 traverse = max(tmin.xx, tmin.yz);
                    float traverselow = max(traverse.x, traverse.y);
                    traverse = min(tmax.xx, tmax.yz);
                    float traversehi = min(traverse.x, traverse.y);

                    if(traversehi > max(traverselow, 0.0) && candidate_id != ignore_id) {
                        rayQueryGenerateIntersectionEXT(ray_query, traverselow);
                        if(traverselow < packed_data[0][0]) {

                            vec3 boxctr = (candidate_v.min + candidate_v.max) / 2.0;

                            vec3 box_hit = boxctr - (candidate_object_space_origin + (traverselow * candidate_object_space_direction));
                            box_hit /= (candidate_v.max - candidate_v.min);
                            vec3 box_intersect_normal = -box_hit / max(max(abs(box_hit.x), abs(box_hit.y)), abs(box_hit.z));
                            box_intersect_normal = clamp(box_intersect_normal, vec3(-1.0), vec3(1.0));
                            box_intersect_normal = normalize(trunc(box_intersect_normal * 1.00001f));

                            // recall that the normal transformation matrix is the transpose of the inverse of the object to world matrix
                            mat4x3 objectToWorldInverse = rayQueryGetIntersectionWorldToObjectEXT(ray_query, false);
                            
                            vec3 normal = normalize((box_intersect_normal * objectToWorldInverse).xyz);

                            vec3 tangent;
                            if(cross(normal, vec3(0.0, 1.0, 0.0)) == vec3(0.0)) {
                                tangent = vec3(1.0, 0.0, 0.0);
                            } else {
                                tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
                            }

                            packed_data = mat4x3(
                                vec3(traverselow, traversehi, 0.0),
                                vec3(uintBitsToFloat(candidate_instance_index), uintBitsToFloat(candidate_prim_index), 0.0),
                                normal,
                                tangent
                            );
                        }
                    }
                }
            }

            // if miss return miss
            if(rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionGeneratedEXT) {
                return IntersectionInfo(
                    true,
                    IntersectionCoordinateSystem(
                        vec3(0.0),
                        vec3(0.0),
                        vec3(0.0)
                    ),
                    0.0,
                    0.0,
                    0,
                    0
                );
            }

            float tmin = packed_data[0][0];
            float tmax = packed_data[0][1];
            uint instance_index = floatBitsToUint(packed_data[1][0]);
            uint primitive_index = floatBitsToUint(packed_data[1][1]);

            vec3 normal = packed_data[2];
            vec3 tangent = packed_data[3];
            vec3 bitangent = cross(normal, tangent);

            return IntersectionInfo(
                false,
                IntersectionCoordinateSystem(
                    normal,
                    tangent,
                    bitangent
                ),
                tmin,
                tmax,
                instance_index,
                primitive_index
            );
        }

        struct BounceInfo {
            vec3 emissivity;
            vec3 reflectivity;
            bool miss;
            vec3 new_origin;
            vec3 new_direction;
            float scatter_pdf_over_ray_pdf;
        };

        BounceInfo doBounce(uint current_bounce, vec3 origin, vec3 direction, IntersectionInfo info, uint seed) {
            if(info.miss) {
                vec3 sky_emissivity = vec3(5.0);
                vec3 sky_reflectivity = vec3(0.0);
                return BounceInfo(
                    sky_emissivity,
                    sky_reflectivity,
                    // miss, so the ray is done
                    true,
                    vec3(0.0),
                    vec3(0.0),
                    1.0
                );
            }

            InstanceData instance = instance_data[info.instance_index];
            GaussianSplat gsplat = GaussianSplat(instance.gaussian_splat_buffer_addr)[info.primitive_index];

            mat3 R = quat_to_mat3(gsplat.rot);
            mat3 S = mat3(
                vec3(gsplat.scale.x, 0.0, 0.0),
                vec3(0.0, gsplat.scale.y, 0.0),
                vec3(0.0, 0.0, gsplat.scale.z)
            );

            vec3 new_origin = origin + direction * info.tmin;
            vec3 new_direction;

            float scatter_pdf_over_ray_pdf;

            vec3 reflectivity = gsplat.color;
            float opacity = 0.1;
            vec3 emissivity = vec3(100*float(info.instance_index == 0));

            // decide whether to do specular (0), transmissive (1), or lambertian (2) scattering
            float scatter_kind_rand = murmur3_finalizef(murmur3_combine(seed, 0));

            if (scatter_kind_rand < opacity) {
                // lambertian scattering
                reflectivity = reflectivity / M_PI;

                // cosine weighted hemisphere sample
                new_direction = alignedCosineWeightedSampleHemisphere(
                    // random uv
                    vec2(
                        murmur3_finalizef(murmur3_combine(seed, 1)),
                        murmur3_finalizef(murmur3_combine(seed, 2))
                    ),
                    // align it with the normal of the object we hit
                    info.hit_coords
                );

                // for lambertian surfaces, the scatter pdf and the ray sampling pdf are the same
                // see here: https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html#lightscattering/thescatteringpdf
                scatter_pdf_over_ray_pdf = 1.0;
            } else {
                // transmissive scattering
                scatter_pdf_over_ray_pdf = 1.0;

                new_direction = direction;
                new_origin = origin + direction * info.tmax;
                reflectivity = vec3(1.0);
                emissivity = vec3(0.0);
            }

            // compute data for this bounce
            return BounceInfo(
                emissivity,
                reflectivity,
                false,
                new_origin,
                new_direction,
                scatter_pdf_over_ray_pdf
            );
        }

        vec2 screen_to_uv(uvec2 screen, uvec2 screen_size) {
            return 2*vec2(screen)/vec2(screen_size) - 1.0;
        }    

        const uint SAMPLES_PER_PIXEL = 8;
        const uint MAX_BOUNCES = 8;

        void main() {
            Camera camera = push_constants.camera;
            if(gl_GlobalInvocationID.x >= camera.screen_size.x || gl_GlobalInvocationID.y >= camera.screen_size.y) {
                return;
            }
    
            uint pixel_seed = murmur3_combine(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
            pixel_seed = murmur3_combine(pixel_seed, push_constants.frame);
    
            vec3 bounce_emissivity[MAX_BOUNCES];
            vec3 bounce_reflectivity[MAX_BOUNCES];
            float bounce_scatter_pdf_over_ray_pdf[MAX_BOUNCES];
    
            vec3 color = vec3(0.0);
            for (uint sample_id = 0; sample_id < SAMPLES_PER_PIXEL; sample_id++) {
                uint sample_seed = murmur3_combine(pixel_seed, sample_id);
    
                // initial ray origin and direction
                vec2 uv = screen_to_uv(gl_GlobalInvocationID.xy, camera.screen_size);
                float aspect = float(camera.screen_size.x) / float(camera.screen_size.y);
    
                vec3 origin = camera.eye;
                vec2 jitter = 0.0*vec2(
                    (1.0/camera.screen_size.x)*(murmur3_finalizef(murmur3_combine(sample_seed, 0))-0.5),
                    (1.0/camera.screen_size.y)*(murmur3_finalizef(murmur3_combine(sample_seed, 1))-0.5)
                );
                vec3 direction = normalize((uv.x + jitter.x) * camera.right * aspect + (uv.y + jitter.y) * camera.up + camera.front);
    
                uint current_bounce;
                for (current_bounce = 0; current_bounce < MAX_BOUNCES; current_bounce++) {
                    IntersectionInfo intersection_info = getIntersectionInfo(origin, direction, 0xFFFFFFFF, 0xFFFFFFFF);
                    BounceInfo bounce_info = doBounce(current_bounce, origin, direction, intersection_info, murmur3_combine(sample_seed, current_bounce));
                    bounce_emissivity[current_bounce] = bounce_info.emissivity;
                    bounce_reflectivity[current_bounce] = bounce_info.reflectivity;
                    bounce_scatter_pdf_over_ray_pdf[current_bounce] = bounce_info.scatter_pdf_over_ray_pdf;
    
                    if(bounce_info.miss) {
                        current_bounce++;
                        break;
                    }
    
                    origin = bounce_info.new_origin;
                    direction = bounce_info.new_direction;
                }
                
                // compute the color for this sample
                vec3 sample_color = vec3(0.0);
                for(int i = int(current_bounce)-1; i >= 0; i--) {
                    sample_color = bounce_emissivity[i] + (sample_color * bounce_reflectivity[i] * bounce_scatter_pdf_over_ray_pdf[i]); 
                }
                color += sample_color;
            }
        
    
            // average the samples
            vec3 pixel_color = (1.0*color) / float(SAMPLES_PER_PIXEL);
            out_color[gl_GlobalInvocationID.y*camera.screen_size.x + gl_GlobalInvocationID.x] = u8vec4(pixel_color.zyx*255, 255);
        }
    ",
}

