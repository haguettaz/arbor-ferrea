#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2 {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }

    /// Translates the pose by adding dx and dy.
    pub fn translate(&mut self, dx: f64, dy: f64) {
        self.x = self.x + dx;
        self.y = self.y + dy;
    }

    /// Rotates the pose by adding dtheta.
    pub fn rotate(&mut self, dtheta: f64) {
        self.theta = self.theta + dtheta;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub theta: f64,
    pub phi: f64,
    pub psi: f64,
}
