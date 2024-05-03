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
            vec3 position;
            uint t;
            vec2 uv;
        };    

        layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer GaussianSplat {
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

        vec3[3] triangleTransform(mat4x3 transform, vec3[3] tri) {
            return vec3[3](
                transform * vec4(tri[0], 1.0),
                transform * vec4(tri[1], 1.0),
                transform * vec4(tri[2], 1.0)
            );
        }    

        struct IntersectionCoordinateSystem {
            vec3 normal;
            vec3 tangent;
            vec3 bitangent;
        };

            
        IntersectionCoordinateSystem localCoordinateSystem(vec3[3] tri) {
            vec3 v0_1 = tri[1] - tri[0];
            vec3 v0_2 = tri[2] - tri[0];
            vec3 normal = cross(v0_1, v0_2);
            vec3 tangent = v0_1;
            vec3 bitangent = cross(normal, tangent);
            
            return IntersectionCoordinateSystem(
                normalize(normal),
                normalize(tangent),
                normalize(bitangent)
            );
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

        // returns a vector sampled from the hemisphere defined around the coordinate system defined by normal, tangent, and bitangent
        // normal, tangent and bitangent form a right handed coordinate system 
        vec3 alignedCosineWeightedSampleHemisphere(vec2 uv, IntersectionCoordinateSystem ics) {
            vec3 hemsam = cosineWeightedSampleHemisphere(uv);
            return normalize(hemsam.x * ics.tangent + hemsam.y * ics.normal + hemsam.z * ics.bitangent);
        }

        struct IntersectionInfo {
            bool miss;
            uint instance_index;
            uint prim_index;
            vec2 bary;
        };
    
        IntersectionInfo getIntersectionInfo(vec3 origin, vec3 direction) {
            const float t_min = 0.001;
            const float t_max = 1000.0;
            rayQueryEXT ray_query;
            rayQueryInitializeEXT(
                ray_query,
                top_level_acceleration_structure,
                gl_RayFlagsNoneEXT,//gl_RayFlagsCullBackFacingTrianglesEXT,
                0xFF,
                origin,
                t_min,
                direction,
                t_max
            );
    
            // trace ray
            while (rayQueryProceedEXT(ray_query));
            
            // if miss return miss
            if(rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                return IntersectionInfo(
                    true,
                    0,
                    0,
                    vec2(0.0)
                );
            } else {
                return IntersectionInfo(
                    false,
                    rayQueryGetIntersectionInstanceIdEXT(ray_query, true),
                    rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true),
                    rayQueryGetIntersectionBarycentricsEXT(ray_query, true)
                );
            }
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


            // get barycentric coordinates
            vec3 bary3 = vec3(1.0 - info.bary.x - info.bary.y,  info.bary.x, info.bary.y);
    
            // get the instance data for this instance
            InstanceData id = instance_data[info.instance_index];

            Vertex t0 = Vertex(id.vertex_buffer_addr)[info.prim_index*3 + 0];
            Vertex t1 = Vertex(id.vertex_buffer_addr)[info.prim_index*3 + 1];
            Vertex t2 = Vertex(id.vertex_buffer_addr)[info.prim_index*3 + 2];

            vec3 tri_r[3] = {t0.position, t1.position, t2.position};
            vec3 tri[3] = triangleTransform(id.transform, tri_r);        
    
            vec3 intersection_point = tri[0] * bary3.x + tri[1] * bary3.y + tri[2] * bary3.z;
            IntersectionCoordinateSystem ics = localCoordinateSystem(tri);

            vec3 gsplat_position;
            vec3 t_u;
            vec3 t_v;
            if (info.prim_index % 2 == 0) {
                gsplat_position = 0.5 * tri[0] + 0.5 * tri[1];
                t_u = tri[1]-tri[2];
                t_v = tri[0]-tri[2];
            } else {
                gsplat_position = 0.5 * tri[0] + 0.5 * tri[2];
                t_u = tri[1]-tri[0];
                t_v = tri[1]-tri[2];
            }

            GaussianSplat gsplat = GaussianSplat(id.gaussian_splat_buffer_addr)[info.prim_index/6];

            // vector from gaussian center to the hit point
            vec3 hit_delta = intersection_point - gsplat_position;

            // project the hit delta onto t_u and t_v
            float hit_delta_proj_u = 4*dot(hit_delta, t_u) / dot(t_u, t_u);
            float hit_delta_proj_v = 4*dot(hit_delta, t_v) / dot(t_v, t_v);

            // get the gaussian value at the point
            float gaussian_value = exp(-.5 * (hit_delta_proj_u*hit_delta_proj_u + hit_delta_proj_v*hit_delta_proj_v));

            vec3 new_origin = intersection_point;
            vec3 new_direction;

            float scatter_pdf_over_ray_pdf;

            vec3 reflectivity = gsplat.color;
            float opacity =  float(gaussian_value > 0.1);
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
                    ics
                );

                // for lambertian surfaces, the scatter pdf and the ray sampling pdf are the same
                // see here: https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html#lightscattering/thescatteringpdf
                scatter_pdf_over_ray_pdf = 1.0;
            } else {
                // transmissive scattering
                scatter_pdf_over_ray_pdf = 1.0;

                new_direction = direction;
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

        const uint SAMPLES_PER_PIXEL = 1;
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
                    IntersectionInfo intersection_info = getIntersectionInfo(origin, direction);
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

