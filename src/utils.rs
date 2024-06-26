use nalgebra::{Point2, Point3, Quaternion, Vector2, Vector3};

use crate::render_system::vertex::Vertex3D;

pub fn flat_polyline(points: Vec<Point3<f32>>, width: f32, color: [f32; 3]) -> Vec<Vertex3D> {
    let normals: Vec<Vector3<f32>> = std::iter::repeat([0.0, 1.0, 0.0].into())
        .take(points.len())
        .collect();
    let width: Vec<f32> = std::iter::repeat(width).take(points.len()).collect();
    let colors = std::iter::repeat(color).take(points.len() - 1).collect();
    polyline(points, normals, width, colors)
}

pub fn polyline(
    points: Vec<Point3<f32>>,
    normals: Vec<Vector3<f32>>,
    width: Vec<f32>,
    colors: Vec<[f32; 3]>,
) -> Vec<Vertex3D> {
    assert!(points.len() > 1, "not enough points");
    assert!(
        points.len() == normals.len(),
        "there must be exactly one normal per point"
    );
    assert!(
        points.len() == width.len(),
        "there must be exactly one width per point"
    );
    assert!(
        points.len() - 1 == colors.len(),
        "there must be exactly one color per line segment"
    );
    // find the vector of each line segment
    let dposition_per_segment: Vec<Vector3<f32>> = points.windows(2).map(|w| w[1] - w[0]).collect();

    // dposition_per_points[0] = dposition_per_segment[0] and dposition_per_points[n] = dposition_per_segment[n-1], but it is the average of the two for the points in between
    let dposition_per_points: Vec<Vector3<f32>> = {
        let mut dposition_per_points = Vec::new();
        dposition_per_points.push(dposition_per_segment[0]);
        for i in 1..dposition_per_segment.len() {
            dposition_per_points
                .push((dposition_per_segment[i - 1] + dposition_per_segment[i]).normalize());
        }
        dposition_per_points.push(dposition_per_segment[dposition_per_segment.len() - 1]);
        dposition_per_points
    };

    // find the cross vectors (along which the width will be applied)
    let cross_vectors: Vec<Vector3<f32>> = dposition_per_points
        .iter()
        .zip(normals.iter())
        .map(|(&v, n)| v.cross(n).normalize())
        .collect();

    // find the left and right points
    let left_points: Vec<Point3<f32>> = cross_vectors
        .iter()
        .zip(width.iter())
        .zip(points.iter())
        .map(|((v, &w), p)| p - v * w)
        .collect();

    let right_points: Vec<Point3<f32>> = cross_vectors
        .iter()
        .zip(width.iter())
        .zip(points.iter())
        .map(|((v, &w), p)| p + v * w)
        .collect();

    let vertexes: Vec<Vertex3D> = std::iter::zip(left_points.windows(2), right_points.windows(2))
        .zip(colors)
        .flat_map(|((l, r), color)| {
            vec![
                Vertex3D::new(r[0].into(), color),
                Vertex3D::new(l[1].into(), color),
                Vertex3D::new(l[0].into(), color),
                Vertex3D::new(r[1].into(), color),
                Vertex3D::new(l[1].into(), color),
                Vertex3D::new(r[0].into(), color),
            ]
        })
        .collect();
    vertexes
}

pub fn xy_quad(loc: Point3<f32>, t_u: Vector3<f32>, t_v: Vector3<f32>, t: u32) -> [Vertex3D; 6] {
    let v00 = loc - t_u * 0.5 - t_v * 0.5;
    let v10 = loc + t_u * 0.5 - t_v * 0.5;
    let v01 = loc - t_u * 0.5 + t_v * 0.5;
    let v11 = loc + t_u * 0.5 + t_v * 0.5;
    [
        Vertex3D::new2(v01.into(), t, [0.0, 0.0]),
        Vertex3D::new2(v10.into(), t, [1.0, 1.0]),
        Vertex3D::new2(v00.into(), t, [0.0, 1.0]),
        Vertex3D::new2(v01.into(), t, [0.0, 0.0]),
        Vertex3D::new2(v11.into(), t, [1.0, 0.0]),
        Vertex3D::new2(v10.into(), t, [1.0, 1.0]),
    ]
}

pub fn cuboid(loc: Point3<f32>, dims: Vector3<f32>) -> Vec<Vertex3D> {
    let fx = loc[0] - 0.5 * dims[0];
    let fy = loc[1] - 0.5 * dims[1];
    let fz = loc[2] - 0.5 * dims[2];

    let v000 = [fx + 0.0, fy + 0.0, fz + 0.0];
    let v100 = [fx + dims[0], fy + 0.0, fz + 0.0];
    let v001 = [fx + 0.0, fy + 0.0, fz + dims[2]];
    let v101 = [fx + dims[0], fy + 0.0, fz + dims[2]];
    let v010 = [fx + 0.0, fy + dims[1], fz + 0.0];
    let v110 = [fx + dims[0], fy + dims[1], fz + 0.0];
    let v011 = [fx + 0.0, fy + dims[1], fz + dims[2]];
    let v111 = [fx + dims[0], fy + dims[1], fz + dims[2]];

    let mut vertexes = vec![];

    // left face
    {
        let t = 0;
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v000, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v011, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
    }

    // right face
    {
        let t = 1;
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v100, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v111, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
    }

    // lower face
    {
        let t = 2;
        vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
    }

    // upper face
    {
        let t = 3;
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
    }

    // back face
    {
        let t = 4;
        vertexes.push(Vertex3D::new2(v010, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v000, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v010, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v110, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 1.0]));
    }

    // front face
    {
        let t = 5;
        vertexes.push(Vertex3D::new2(v001, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v101, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v111, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 0.0]));
    }

    vertexes
}

pub fn unitcube() -> Vec<Vertex3D> {
    cuboid(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0))
}

// get axis aligned bounding box
pub fn get_aabb(obj: &[Vertex3D]) -> Vector3<f32> {
    let mut min = Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX);
    let mut max = Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN);
    for v in obj.iter() {
        if v.position[0] < min[0] {
            min[0] = v.position[0];
        }
        if v.position[1] < min[1] {
            min[1] = v.position[1];
        }
        if v.position[2] < min[2] {
            min[2] = v.position[2];
        }
        if v.position[0] > max[0] {
            max[0] = v.position[0];
        }
        if v.position[1] > max[1] {
            max[1] = v.position[1];
        }
        if v.position[2] > max[2] {
            max[2] = v.position[2];
        }
    }
    max - min
}

pub fn get_normalized_mouse_coords(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let trackball_radius = extent[0].min(extent[1]) as f32;
    let center = Vector2::new(extent[0] as f32 / 2.0, extent[1] as f32 / 2.0);
    return (e - center) / trackball_radius;
}

pub fn screen_to_uv(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let x = e[0] / extent[0] as f32;
    let y = e[1] / extent[1] as f32;
    Point2::new(2.0 * x - 1.0, 2.0 * y - 1.0)
}

#[derive(Clone)]
pub struct PointCloudPoint {
    pub position: Point3<f32>,
    pub scale: Vector3<f32>,
    pub rot: Quaternion<f32>,
    pub color: [f32; 3],
    pub opacity: f32,
}

impl ply_rs::ply::PropertyAccess for PointCloudPoint {
    fn new() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rot: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            color: [0.0, 0.0, 0.0],
            opacity: 1.0,
        }
    }

    fn set_property(&mut self, property: String, value: ply_rs::ply::Property) {
        fn as_float(property: ply_rs::ply::Property) -> f32 {
            match property {
                ply_rs::ply::Property::Float(v) => v,
                _ => panic!("expected float"),
            }
        }

        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }

        match property.as_str() {
            "x" => self.position[0] = as_float(value),
            "y" => self.position[1] = as_float(value),
            "z" => self.position[2] = as_float(value),
            "scale_0" => self.scale[0] = as_float(value).exp(),
            "scale_1" => self.scale[1] = as_float(value).exp(),
            "scale_2" => self.scale[2] = as_float(value).exp(),
            // w x y z
            "rot_0" => self.rot[3] = as_float(value),
            "rot_1" => self.rot[0] = as_float(value),
            "rot_2" => self.rot[1] = as_float(value),
            "rot_3" => self.rot[2] = as_float(value),
            "f_dc_0" => self.color[0] = sigmoid(as_float(value)),
            "f_dc_1" => self.color[1] = sigmoid(as_float(value)),
            "f_dc_2" => self.color[2] = sigmoid(as_float(value)),
            "opacity" => self.opacity = sigmoid(as_float(value)),
            _ => {}
        }
    }
}
