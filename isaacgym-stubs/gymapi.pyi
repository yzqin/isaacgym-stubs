from typing import Any, ClassVar, Dict, List, Tuple

from typing import overload
import numpy

AXIS_ALL: int = 63
AXIS_NONE: int = 0
AXIS_ROTATION: int = 56
AXIS_SWING_1: int = 16
AXIS_SWING_2: int = 32
AXIS_TRANSLATION: int = 7
AXIS_TWIST: int = 8
AXIS_X: int = 1
AXIS_Y: int = 2
AXIS_Z: int = 4
CC_ALL_SUBSTEPS: ContactCollection
CC_LAST_SUBSTEP: ContactCollection
CC_NEVER: ContactCollection
COMPUTE_PER_FACE: MeshNormalMode
COMPUTE_PER_VERTEX: MeshNormalMode
DEFAULT_VIEWER_HEIGHT: int = 900
DEFAULT_VIEWER_WIDTH: int = 1600
DOF_INVALID: DofType
DOF_MODE_EFFORT: DofDriveMode
DOF_MODE_NONE: DofDriveMode
DOF_MODE_POS: DofDriveMode
DOF_MODE_VEL: DofDriveMode
DOF_ROTATION: DofType
DOF_TRANSLATION: DofType
DOMAIN_ACTOR: IndexDomain
DOMAIN_ENV: IndexDomain
DOMAIN_SIM: IndexDomain
DTYPE_FLOAT32: TensorDataType
DTYPE_INT16: TensorDataType
DTYPE_UINT32: TensorDataType
DTYPE_UINT64: TensorDataType
DTYPE_UINT8: TensorDataType
ENV_SPACE: CoordinateSpace
FOLLOW_POSITION: CameraFollowMode
FOLLOW_TRANSFORM: CameraFollowMode
FROM_ASSET: MeshNormalMode
GLOBAL_SPACE: CoordinateSpace
IMAGE_COLOR: ImageType
IMAGE_DEPTH: ImageType
IMAGE_OPTICAL_FLOW: ImageType
IMAGE_SEGMENTATION: ImageType
INVALID_HANDLE: int = -1
JOINT_BALL: JointType
JOINT_FIXED: JointType
JOINT_FLOATING: JointType
JOINT_INVALID: JointType
JOINT_PLANAR: JointType
JOINT_PRISMATIC: JointType
JOINT_REVOLUTE: JointType
KEY_0: KeyboardInput
KEY_1: KeyboardInput
KEY_2: KeyboardInput
KEY_3: KeyboardInput
KEY_4: KeyboardInput
KEY_5: KeyboardInput
KEY_6: KeyboardInput
KEY_7: KeyboardInput
KEY_8: KeyboardInput
KEY_9: KeyboardInput
KEY_A: KeyboardInput
KEY_APOSTROPHE: KeyboardInput
KEY_B: KeyboardInput
KEY_BACKSLASH: KeyboardInput
KEY_BACKSPACE: KeyboardInput
KEY_C: KeyboardInput
KEY_CAPS_LOCK: KeyboardInput
KEY_COMMA: KeyboardInput
KEY_D: KeyboardInput
KEY_DEL: KeyboardInput
KEY_DOWN: KeyboardInput
KEY_E: KeyboardInput
KEY_END: KeyboardInput
KEY_ENTER: KeyboardInput
KEY_EQUAL: KeyboardInput
KEY_ESCAPE: KeyboardInput
KEY_F: KeyboardInput
KEY_F1: KeyboardInput
KEY_F10: KeyboardInput
KEY_F11: KeyboardInput
KEY_F12: KeyboardInput
KEY_F2: KeyboardInput
KEY_F3: KeyboardInput
KEY_F4: KeyboardInput
KEY_F5: KeyboardInput
KEY_F6: KeyboardInput
KEY_F7: KeyboardInput
KEY_F8: KeyboardInput
KEY_F9: KeyboardInput
KEY_G: KeyboardInput
KEY_GRAVE_ACCENT: KeyboardInput
KEY_H: KeyboardInput
KEY_HOME: KeyboardInput
KEY_I: KeyboardInput
KEY_INSERT: KeyboardInput
KEY_J: KeyboardInput
KEY_K: KeyboardInput
KEY_L: KeyboardInput
KEY_LEFT: KeyboardInput
KEY_LEFT_ALT: KeyboardInput
KEY_LEFT_BRACKET: KeyboardInput
KEY_LEFT_CONTROL: KeyboardInput
KEY_LEFT_SHIFT: KeyboardInput
KEY_LEFT_SUPER: KeyboardInput
KEY_M: KeyboardInput
KEY_MENU: KeyboardInput
KEY_MINUS: KeyboardInput
KEY_N: KeyboardInput
KEY_NUMPAD_0: KeyboardInput
KEY_NUMPAD_1: KeyboardInput
KEY_NUMPAD_2: KeyboardInput
KEY_NUMPAD_3: KeyboardInput
KEY_NUMPAD_4: KeyboardInput
KEY_NUMPAD_5: KeyboardInput
KEY_NUMPAD_6: KeyboardInput
KEY_NUMPAD_7: KeyboardInput
KEY_NUMPAD_8: KeyboardInput
KEY_NUMPAD_9: KeyboardInput
KEY_NUMPAD_ADD: KeyboardInput
KEY_NUMPAD_DEL: KeyboardInput
KEY_NUMPAD_DIVIDE: KeyboardInput
KEY_NUMPAD_ENTER: KeyboardInput
KEY_NUMPAD_EQUAL: KeyboardInput
KEY_NUMPAD_MULTIPLY: KeyboardInput
KEY_NUMPAD_SUBTRACT: KeyboardInput
KEY_NUM_LOCK: KeyboardInput
KEY_O: KeyboardInput
KEY_P: KeyboardInput
KEY_PAGE_DOWN: KeyboardInput
KEY_PAGE_UP: KeyboardInput
KEY_PAUSE: KeyboardInput
KEY_PERIOD: KeyboardInput
KEY_PRINT_SCREEN: KeyboardInput
KEY_Q: KeyboardInput
KEY_R: KeyboardInput
KEY_RIGHT: KeyboardInput
KEY_RIGHT_ALT: KeyboardInput
KEY_RIGHT_BRACKET: KeyboardInput
KEY_RIGHT_CONTROL: KeyboardInput
KEY_RIGHT_SHIFT: KeyboardInput
KEY_RIGHT_SUPER: KeyboardInput
KEY_S: KeyboardInput
KEY_SCROLL_LOCK: KeyboardInput
KEY_SEMICOLON: KeyboardInput
KEY_SLASH: KeyboardInput
KEY_SPACE: KeyboardInput
KEY_T: KeyboardInput
KEY_TAB: KeyboardInput
KEY_U: KeyboardInput
KEY_UP: KeyboardInput
KEY_V: KeyboardInput
KEY_W: KeyboardInput
KEY_X: KeyboardInput
KEY_Y: KeyboardInput
KEY_Z: KeyboardInput
LOCAL_SPACE: CoordinateSpace
MAT_COROTATIONAL: SoftMaterialType
MAT_NEOHOOKEAN: SoftMaterialType
MESH_COLLISION: MeshType
MESH_NONE: MeshType
MESH_VISUAL: MeshType
MESH_VISUAL_AND_COLLISION: MeshType
MOUSE_BACK_BUTTON: MouseInput
MOUSE_FORWARD_BUTTON: MouseInput
MOUSE_LEFT_BUTTON: MouseInput
MOUSE_MIDDLE_BUTTON: MouseInput
MOUSE_MOVE_DOWN: MouseInput
MOUSE_MOVE_LEFT: MouseInput
MOUSE_MOVE_RIGHT: MouseInput
MOUSE_MOVE_UP: MouseInput
MOUSE_RIGHT_BUTTON: MouseInput
MOUSE_SCROLL_DOWN: MouseInput
MOUSE_SCROLL_LEFT: MouseInput
MOUSE_SCROLL_RIGHT: MouseInput
MOUSE_SCROLL_UP: MouseInput
RIGID_BODY_DISABLE_GRAVITY: int = 1
RIGID_BODY_DISABLE_SIMULATION: int = 2
RIGID_BODY_ENABLE_GYROSCOPIC_FORCES: int
RIGID_BODY_NONE: int = 0
SIM_FLEX: SimType
SIM_PHYSX: SimType
STATE_ALL: int = 3
STATE_NONE: int = 0
STATE_POS: int = 1
STATE_VEL: int = 2
TENDON_FIXED: TendonType
TENDON_SPATIAL: TendonType
UP_AXIS_Y: UpAxis
UP_AXIS_Z: UpAxis
USD_MATERIAL_DISPLAY_COLOR: UsdMaterialMode
USD_MATERIAL_MDL: UsdMaterialMode
USD_MATERIAL_PREVIEW_SURFACE: UsdMaterialMode


class ActionEvent:
    ''' class isaacgym.gymapi.ActionEvent
    '''

    def __init__(self) -> None: ...

    @property
    def action(self) -> str: ...
    ''' property action
    '''


    @property
    def value(self) -> float: ...
    ''' property value
    '''



class ActuatorProperties:
    control_limited: bool
    force_limited: bool
    kp: float
    kv: float
    lower_control_limit: float
    lower_force_limit: float
    motor_effort: float
    upper_control_limit: float
    upper_force_limit: float

    def __init__(self) -> None: ...

    @property
    def type(self) -> Any: ...


class Asset:
    def __init__(self, *args, **kwargs) -> None: ...


class AssetOptions:
    ''' class isaacgym.gymapi.AssetOptions
    Defines a set of properties for assets imported into Gym.
    '''

    angular_damping: float  # property angular_damping Angular velocity damping for rigid bodies
    armature: float  # property armature The value added to the diagonal elements of inertia tensors for all of the asset's rigid bodies/links. Could improve simulation stability
    collapse_fixed_joints: bool  # property collapse_fixed_joints Merge links that are connected by fixed joints.
    convex_decomposition_from_submeshes: bool  # property convex_decomposition_from_submeshes Whether to treat submeshes in the mesh as the convex decomposition of the mesh. Default False.
    default_dof_drive_mode: int  # property default_dof_drive_mode Default mode used to actuate Asset joints. See isaacgym.gymapi.DriveModeFlags.
    density: float  # property density Default density parameter used for calculating mass and inertia tensor when no mass and inertia data are provided, in $kg/m^3$.
    disable_gravity: bool  # property disable_gravity Disables gravity for asset.
    enable_gyroscopic_forces: bool  # property enable_gyroscopic_forces Enable gyroscopic forces (PhysX only).
    fix_base_link: bool  # property fix_base_link Set Asset base to a fixed placement upon import.
    flip_visual_attachments: bool  # property flip_visual_attachments Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
    linear_damping: float  # property linear_damping Linear velocity damping for rigid bodies.
    max_angular_velocity: float  # property max_angular_velocity Maximum angular velocity for rigid bodies. In $rad/s$.
    max_linear_velocity: float  # property max_linear_velocity Maximum linear velocity for rigid bodies. In $m/s$.
    mesh_normal_mode: MeshNormalMode  # property mesh_normal_mode How to load normals for the meshes in the asset. One of FROM_ASSET, COMPUTE_PER_VERTEX, or COMPUTE_PER_FACE. Defaults to FROM_ASSET, falls back to COMPUTE_PER_VERTEX if normals not fully specified in mesh.
    min_particle_mass: float  # property min_particle_mass Minimum mass for particles in soft bodies, in Kg
    override_com: bool  # property override_com Whether to compute the center of mass from geometry and override values given in the original asset.
    override_inertia: bool  # property override_inertia Whether to compute the inertia tensor from geometry and override values given in the original asset.
    replace_cylinder_with_capsule: bool  # property replace_cylinder_with_capsule flag to replace Cylinders with capsules for additional performance.
    slices_per_cylinder: int  # property slices_per_cylinder Number of faces on generated cylinder mesh, excluding top and bottom.
    tendon_limit_stiffness: float  # property tendon_limit_stiffness Default tendon limit stiffness. Choose small as the limits are not implicitly solved. Avoid oscillations by setting an apporpriate damping value.
    thickness: float  # property thickness Thickness of the collision shapes. Sets how far objects should come to rest from the surface of this body
    use_mesh_materials: bool  # property use_mesh_materials Whether to use materials loaded from mesh files instead of the materials defined in asset file. Default False.
    use_physx_armature: bool  # property use_physx_armature Use joint space armature instead of links inertia tensor modififcations.
    vhacd_enabled: bool  # property vhacd_enabled Whether convex decomposition is enabled.  Used only with PhysX.  Default False.
    vhacd_params: VhacdParams  # property vhacd_params Convex decomposition parameters.  Used only with PhysX.  If not specified, all triangle meshes will be approximated using a single convex hull.

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class AttractorProperties:
    ''' class isaacgym.gymapi.AttractorProperties
    The Attractor is used to pull a rigid body towards a pose. Each pose axis can be individually selected.
    '''

    axes: int  # property axes Axes to set the attractor, using GymTransformAxesFlags. Multiple axes can be selected using bitwise combination of each axis flag. if axis flag is set to zero, the attractor will be disabled and won't impact in solver computational complexity.
    damping: float  # property damping Damping to be used on attraction solver.
    offset: Transform  # property offset Offset from rigid body origin to set the attractor pose.
    rigid_handle: int  # property rigid_handle Handle to the rigid body to set the attractor to
    stiffness: float  # property stiffness Stiffness to be used on attraction for solver. Stiffness value should be larger than the largest agent kinematic chain stifness
    target: Transform  # property target Target pose to attract to.

    def __init__(self) -> None: ...


class CameraFollowMode:
    ''' class isaacgym.gymapi.CameraFollowMode
    Camera follow mode
    Members:
    
    FOLLOW_POSITION : Camera attached to a rigid body follows only its position
    FOLLOW_TRANSFORM : Camera attached to a rigid body follows its transform (both position and orientation)
    '''

    __members__: ClassVar[dict] = ...  # read-only
    FOLLOW_POSITION: ClassVar[CameraFollowMode] = ...
    FOLLOW_TRANSFORM: ClassVar[CameraFollowMode] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class CameraProperties:
    ''' class isaacgym.gymapi.CameraProperties
    Properties for a camera in Gym
    '''

    enable_tensors: bool  # property enable_tensors CUDA interop buffers will be available only if this is true.
    far_plane: float  # property far_plane distance in world coordinates to far-clipping plane
    height: int  # property height Height of output images in pixels
    horizontal_fov: float  # property horizontal_fov Horizontal field of view in degrees. Vertical field of view is calculated from height to width ratio
    near_plane: float  # property near_plane distance in world coordinate units to near-clipping plane
    supersampling_horizontal: int  # property supersampling_horizontal oversampling factor in the horiziontal/X direction
    supersampling_vertical: int  # property supersampling_vertical oversampling factor in the vertical/Y direction
    use_collision_geometry: bool  # property use_collision_geometry If true, camera renders collision meshes instead of visual meshes
    width: int  # property width Width of output images in pixels

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class ContactCollection:
    ''' class isaacgym.gymapi.ContactCollection
    Contact collection mode.
    Members:
    
    CC_NEVER : Don't collect any contacts (value = 0).
    CC_LAST_SUBSTEP : Collect contacts for last substep only (value = 1).
    CC_ALL_SUBSTEPS : Collect contacts for all substeps (value = 2) (default).
    '''

    __members__: ClassVar[dict] = ...  # read-only
    CC_ALL_SUBSTEPS: ClassVar[ContactCollection] = ...
    CC_LAST_SUBSTEP: ClassVar[ContactCollection] = ...
    CC_NEVER: ClassVar[ContactCollection] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class CoordinateSpace:
    ''' class isaacgym.gymapi.CoordinateSpace
    Coordinate system for positions.
    Members:
    
    ENV_SPACE
    LOCAL_SPACE
    GLOBAL_SPACE
    '''

    __members__: ClassVar[dict] = ...  # read-only
    ENV_SPACE: ClassVar[CoordinateSpace] = ...
    GLOBAL_SPACE: ClassVar[CoordinateSpace] = ...
    LOCAL_SPACE: ClassVar[CoordinateSpace] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class DofDriveMode:
    ''' class isaacgym.gymapi.DofDriveMode
    Possible drive modes used to control actor DoFs.  A DoF that is set to a specific drive mode will ignore drive commands for other modes.
    Members:
    
    DOF_MODE_NONE : The DOF is free to move without any controls.
    DOF_MODE_POS : The DOF will respond to position target commands.
    DOF_MODE_VEL : The DOF will respond to velocity target commands.
    DOF_MODE_EFFORT : The DOF will respond to effort (force or torque) commands.
    '''

    __members__: ClassVar[dict] = ...  # read-only
    DOF_MODE_EFFORT: ClassVar[DofDriveMode] = ...
    DOF_MODE_NONE: ClassVar[DofDriveMode] = ...
    DOF_MODE_POS: ClassVar[DofDriveMode] = ...
    DOF_MODE_VEL: ClassVar[DofDriveMode] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class DofFrame:
    ''' class isaacgym.gymapi.DofFrame
    Frame of a Degree of Freedom
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('origin', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]), ('axis', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')])])
    axis: Vec3  # property axis direction for the DOF action
    origin: Vec3  # property origin position in environment for the DOF

    def __init__(self, origin: Vec3 = ..., axis: Vec3 = ...) -> None: ...

    def from_buffer(self, *args, **kwargs) -> Any: ...
    ''' static from_buffer(arg0: buffer) -> object
    '''


    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class DofState:
    ''' class isaacgym.gymapi.DofState
    States of a Degree of Freedom in the Asset architecture
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('pos', '<f4'), ('vel', '<f4')])
    pos: float  # property pos DOF position, in radians if it's a revolute DOF, or meters, if it's a prismatic DOF
    vel: float  # property vel DOF velocity, in radians/s if it's a revolute DOF, or m/s, if it's a prismatic DOF

    def __init__(self, *args, **kwargs) -> None: ...


class DofType:
    ''' class isaacgym.gymapi.DofType
    Types of degree of freedom supported by the simulator
    Members:
    
    DOF_INVALID : invalid/unknown/uninitialized DOF type
    DOF_ROTATION : The degrees of freedom correspond to a rotation between bodies
    DOF_TRANSLATION : The degrees of freedom correspond to a translation between bodies.
    '''

    __members__: ClassVar[dict] = ...  # read-only
    DOF_INVALID: ClassVar[DofType] = ...
    DOF_ROTATION: ClassVar[DofType] = ...
    DOF_TRANSLATION: ClassVar[DofType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class Env:
    def __init__(self, *args, **kwargs) -> None: ...


class FlexParams:
    ''' class isaacgym.gymapi.FlexParams
    Simulation parameters used for FleX physics engine
    '''

    contact_regularization: float  # property contact_regularization Distance for soft bodies to maintain against ground planes
    deterministic_mode: bool  # property deterministic_mode Flag to activate deterministic simulation. Flex Newton solver only
    dynamic_friction: float  # property dynamic_friction Coefficient of friction used when colliding against shapes
    friction_mode: int  # property friction_mode Type of friction mode: 0 single friction dir, non-linear cone projection, but can't change direction during linear solve 1 two friction dirs, non-linear cone projection, can change direction during linear solve 2 same as above plus torsional (spinning) friction
    geometric_stiffness: float  # property geometric_stiffness Improves stability of joints by approximating the system Hessian
    max_rigid_contacts: int  # property max_rigid_contacts Max number of rigid body contacts
    max_soft_contacts: int  # property max_soft_contacts Max number of soft body contacts
    num_inner_iterations: int  # property num_inner_iterations Number of inner loop iterations taken by the solver per simulation step. Is used only by Newton solver.
    num_outer_iterations: int  # property num_outer_iterations Number of iterations taken by the solver per simulation step.
    particle_friction: float  # property particle_friction Coefficient of friction used when colliding particles
    relaxation: float  # property relaxation Control the convergence rate of the parallel solver. Values greater than 1 may lead to instability.
    return_contacts: bool  # property return_contacts Read contact information back to CPU
    shape_collision_distance: float  # property shape_collision_distance Distance for soft bodies to maintain against rigid bodies and ground plane
    shape_collision_margin: float  # property shape_collision_margin Distance for rigid bodies at which contacts are generated
    solver_type: int  # property solver_type Type of solver used:  0 = XPBD (GPU) 1 = Newton Jacobi (GPU) 2 = Newton LDLT (CPU) 3 = Newton PCG (CPU) 4 = Newton PCG (GPU) 5 = Newton PCR (GPU) 6 = Newton Gauss Seidel (CPU) 7 = Newton NNCG (GPU)
    static_friction: float  # property static_friction Coefficient of static friction used when colliding against shapes
    warm_start: float  # property warm_start Fraction of the cached Lagrange Multiplier to be used on the next simulation step.

    def __init__(self) -> None: ...


class ForceSensor:
    def __init__(self) -> None: ...

    def get_forces(self) -> SpatialForce: ...

    def get_global_index(self) -> int: ...


class ForceSensorProperties:
    ''' class isaacgym.gymapi.ForceSensorProperties
    Set of properties used for force sensors.
    '''

    enable_constraint_solver_forces: bool  # property enable_constraint_solver_forces Enable to receive forces from constraint solver (default = True).
    enable_forward_dynamics_forces: bool  # property enable_forward_dynamics_forces Enable to receive forces from forward dynamics (default = True).
    use_world_frame: bool  # property use_world_frame Enable to receive forces in the world rotation frame, otherwise they will be reported in the sensor's local frame (default = False).

    def __init__(self) -> None: ...


class Gym:
    ''' class isaacgym.gymapi.Gym
    '''

    def __init__(self, *args, **kwargs) -> None: ...

    def acquire_actor_root_state_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_actor_root_state_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for Actor root states.  The buffer has shape (num_actors, 13).
    State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for actor root states
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_dof_force_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_dof_force_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for DOF forces. One force value per each DOF in simulation.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for DOF forces
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_dof_state_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_dof_state_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves Degree-of-Freedom state buffer. Buffer has shape (num_dofs, 2).
    Each DOF state contains position and velocity.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for DOF states
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_force_sensor_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_force_sensor_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for force sensors. The buffer has shape (num_force_sensors, 6).
    Each force sensor state has forces (3) and torques (3) data.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for force sensors
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_jacobian_tensor(self, arg0: Sim, arg1: str) -> Tensor: ...
    ''' acquire_jacobian_tensor(self: Gym, arg0: Sim, arg1: str) -> Tensor
    Retrieves buffer information for Jacobian
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (str) - Name of Actor
    
    
    Returns:
    GymTensor object containing buffer for Jacobian
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_mass_matrix_tensor(self, arg0: Sim, arg1: str) -> Tensor: ...
    ''' acquire_mass_matrix_tensor(self: Gym, arg0: Sim, arg1: str) -> Tensor
    Retrieves buffer for Mass matrix
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (str) - Name of Actor
    
    
    Returns:
    GymTensor object containing buffer for Mass Matrix
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_net_contact_force_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_net_contact_force_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for net contract forces. The buffer has shape (num_rigid_bodies, 3).
    Each contact force state contains one value for each X, Y, Z axis.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for net contact forces
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_particle_state_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_particle_state_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for particle states. Flex backend only.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for particle states
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_pneumatic_pressure_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_pneumatic_pressure_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for penumatic pressure states. Flex backend only.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for pneumatic pressure
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_pneumatic_target_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_pneumatic_target_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for pneumatic targets. Flex backend only.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for pneumatic targets
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def acquire_rigid_body_state_tensor(self, arg0: Sim) -> Tensor: ...
    ''' acquire_rigid_body_state_tensor(self: Gym, arg0: Sim) -> Tensor
    Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13).
    State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    GymTensor object containing buffer for rigid body states
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def add_ground(self, sim: Sim, params: PlaneParams) -> None: ...
    ''' add_ground(self: Gym, sim: Sim, params: PlaneParams) -> None
    Adds ground plane to simulation.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (isaacgym.gymapi.PlaneParams) - Structure of parameters for ground plane
    '''


    def add_heightfield(self, arg0: Sim, arg1: numpy.ndarray[int16], arg2: HeightFieldParams) -> None: ...
    ''' add_heightfield(self: Gym, arg0: Sim, arg1: numpy.ndarray[int16], arg2: HeightFieldParams) -> None
    Adds ground heightfield to simulation.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (heightSamples) - Height samples as Int16 array. Column-major order.
    param3 (carbongym.gymapi.GymHeightFieldParams) - Structure of parameters for heightfield
    '''


    @overload
    def add_lines(self, arg0: Viewer, arg1: Env, arg2: int, arg3: numpy.ndarray[float32],
                  arg4: numpy.ndarray[float32]) -> None: ...
    ''' add_lines(*args, **kwargs)
    Overloaded function.
    
    add_lines(self: Gym, arg0: Viewer, arg1: Env, arg2: int, arg3: numpy.ndarray[float32], arg4: numpy.ndarray[float32]) -> None
    Adds lines to the viewer. Adds lines that start and end on the specified vertices, with the colors defined.
    Each line is defined by a tuple of 6 floats in the vertices array, organized as {p1.x, p1.y, p1.z, p2.x, p2.y, p2.z}, defined in the environment frame.
    Each color is a tuple of 3 floats ranging from [0,1] representing the {r, g, b} spectrum.
    
    
    Args:param1 (Viewer): Viewer Handle.
    param2 (Env): Environment Handle.
    param3 (int): number of lines to draw
    param4 (array of float): vertices of the lines. Must be of size at least 6*numLines
    param5 (array of float): colors to be applied to the lines. Must be of size at least 3*numLines
    Adds lines to the viewer. Adds lines that start and end on the specified vertices, with the colors defined.
    Each line is defined by a tuple of 6 floats in the vertices array, organized as {p1.x, p1.y, p1.z, p2.x, p2.y, p2.z}, defined in the environment frame.
    Each color is a tuple of 3 floats ranging from [0,1] representing the {r, g, b} spectrum.
    
    
    Args:param1 (Viewer): Viewer Handle.
    param2 (Env): Environment Handle.
    param3 (int): number of lines to draw
    param4 (array of isaacgym.gymapi.Vec3): vertices of the lines. Must be of size at least 2*numLines
    param5 (array of isaacgym.gymapi.Vec3): colors to be applied to the lines. Must be of size at least numLines
    '''


    @overload
    def add_lines(self, arg0: Viewer, arg1: Env, arg2: int, arg3: numpy.ndarray[Vec3],
                  arg4: numpy.ndarray[Vec3]) -> None: ...

    def add_triangle_mesh(self, arg0: Sim, arg1: numpy.ndarray[float32], arg2: numpy.ndarray[uint32],
                          arg3: TriangleMeshParams) -> None: ...
    ''' add_triangle_mesh(self: Gym, arg0: Sim, arg1: numpy.ndarray[float32], arg2: numpy.ndarray[uint32], arg3: TriangleMeshParams) -> None
    Adds ground heightfield to simulation.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (heightSamples) - Height samples as Int16 array. Column-major order.
    param3 (carbongym.gymapi.GymHeightFieldParams) - Structure of parameters for heightfield
    '''


    def apply_actor_dof_efforts(self, arg0: Env, arg1: int, arg2: numpy.ndarray[float32]) -> bool: ...
    ''' apply_actor_dof_efforts(self: Gym, arg0: Env, arg1: int, arg2: numpy.ndarray[float32]) -> bool
    Applies efforts passed as an ordered array to the Degrees of Freedom of an actor.
    If the Degree of Freedom is linear, the effort is a force in Newton.
    If the Degree of Freedom is revolute, the effort is a torque in Nm.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of float) : array containing the efforts for all degrees of freedom of the actor.
    
    
    Returns:
    true if viewer has closed, false otherwise
    
    Return type:
    bool
    '''


    def apply_body_force_at_pos(self, env: Env, rigidHandle: int, force: Vec3, pos: Vec3 = ...,
                                space: CoordinateSpace = ...) -> None: ...
    ''' apply_body_force_at_pos(self: Gym, env: Env, rigidHandle: int, force: Vec3, pos: Vec3 = None, space: CoordinateSpace = CoordinateSpace.ENV_SPACE) -> None
    Applies a force at the given position of the selected body for the immediate timestep, in Newtons.
    If the force is not at the center-of-mass of the body, a torque will be applied in Nm.
    Parameters:
    
    env (Env) - Environment
    rigidHandle (Body) - Rigid body handle
    force (isaacgym.gymapi.Vec3) - force
    pos (isaacgym.gymapi.Vec3) - pos, can be None
    space (isaacgym.gymapi.CoordinateSpace) - coordinate space of force and position vectors
    '''


    def apply_body_forces(self, env: Env, rigidHandle: int, force: Vec3 = ..., torque: Vec3 = ...,
                          space: CoordinateSpace = ...) -> None: ...
    ''' apply_body_forces(self: Gym, env: Env, rigidHandle: int, force: Vec3 = None, torque: Vec3 = None, space: CoordinateSpace = CoordinateSpace.ENV_SPACE) -> None
    Applies a force and/or torque to the selected body for the immediate timestep, in Newtons and Nm respectively.
    The force is applied at the center of mass of the body.
    Parameters:
    
    env (Env) - Environment
    rigidHandle (Body) - Rigid body handle
    force (isaacgym.gymapi.Vec3) - force, can be None
    torque (isaacgym.gymapi.Vec3) - torque, can be None
    space (isaacgym.gymapi.CoordinateSpace) - coordinate space of the force and torque vectors
    '''


    def apply_dof_effort(self, arg0: Env, arg1: int, arg2: float) -> None: ...
    ''' apply_dof_effort(self: Gym, arg0: Env, arg1: int, arg2: float) -> None
    Applies effort on a DOF. If the DOF is prismatic, the effort will be a force in Newtons. If the DOF is revolute, the effort will be a Torque, in Nm.
    See isaacgym.gymapi.Gym.set_dof_actuation_force_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - DOF Handle
    param3 (float) - effort
    '''


    def apply_rigid_body_force_at_pos_tensors(self, sim: Sim, forceTensor: Tensor, posTensor: Tensor = ...,
                                              space: CoordinateSpace = ...) -> bool: ...
    ''' apply_rigid_body_force_at_pos_tensors(self: Gym, sim: Sim, forceTensor: Tensor, posTensor: Tensor = None, space: CoordinateSpace = CoordinateSpace.ENV_SPACE) -> bool
    Applies rigid body forces at given positions for the immediate timestep, in Newtons.
    Parameters:
    
    sim (Sim) - Simulation Handle
    forceTensor (isaacgym.gymapi.Tensor) - Buffer containing forces
    posTensor (isaacgym.gymapi.Tensor) - Buffer containing positions, can be None (if None, forces will be applied at CoM)
    space (isaacgym.gymapi.CoordinateSpace) - Coordinate space of force and torque vectors
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def apply_rigid_body_force_tensors(self, sim: Sim, forceTensor: Tensor = ..., torqueTensor: Tensor = ...,
                                       space: CoordinateSpace = ...) -> bool: ...
    ''' apply_rigid_body_force_tensors(self: Gym, sim: Sim, forceTensor: Tensor = None, torqueTensor: Tensor = None, space: CoordinateSpace = CoordinateSpace.ENV_SPACE) -> bool
    Applies forces and/or torques to rigid bodies for the immediate timestep, in Newtons.
    Parameters:
    
    sim (Sim) - Simulation Handle
    forceTensor (isaacgym.gymapi.Tensor) - Buffer containing forces, can be None
    torqueTensor (isaacgym.gymapi.Tensor) - Buffer containing torques, can be None
    space (isaacgym.gymapi.CoordinateSpace) - Coordinate space of force and torque vectors
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def attach_camera_to_body(self, arg0: int, arg1: Env, arg2: int, arg3: Transform,
                              arg4: CameraFollowMode) -> None: ...
    ''' attach_camera_to_body(self: Gym, arg0: int, arg1: Env, arg2: int, arg3: Transform, arg4: CameraFollowMode) -> None
    Attaches Camera to a rigid body
    Parameters:
    
    param1 (Camera) - Camera Handle.
    param2 (Env) - Environment Handle.
    param3 (Body) - handle to the rigid body
    param4 (isaacgym.gymapi.Transform) - transform from rigid body to camera position
    param5 (isaacgym.gymapi.CameraFollowMode) - Follow mode. see isaacgym.gymapi.CameraFollowMode
    '''


    def attach_sim(self, arg0: int, arg1: int, arg2: SimType, arg3: str, arg4: str) -> Sim: ...
    ''' attach_sim(self: Gym, arg0: int, arg1: int, arg2: SimType, arg3: str, arg4: str) -> Sim
    Attach simulation to USD scene, updates will be saved to USD stage.
    Parameters:
    
    param1 (int) - index of CUDA-enabled GPU to be used for simulation.
    param2 (int) - index of GPU to be used for rendering.
    param3 (isaacgym.gymapi.SimType) - Type of simulation to be used.
    param4 (string) - Path to root directory of USD scene
    param5 (string) - Filename of USD scene
    
    
    Returns:
    Simulation Handle
    
    Return type:
    Sim
    '''


    def begin_aggregate(self, arg0: Env, arg1: int, arg2: int, arg3: bool) -> bool: ...
    ''' begin_aggregate(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: bool) -> bool
    Creates new aggregate group
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (int) - Maximum number of bodies
    param3 (int) - Maximum number of shapes
    param4 (bool) - Flag to enable or disable self collision
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def clear_lines(self, arg0: Viewer) -> None: ...
    ''' clear_lines(self: Gym, arg0: Viewer) -> None
    Clears all lines from the viewer
    Parameters:
    param1 (Viewer) - Viewer Handle.
    '''


    def create_actor(self, env: Env, asset: Asset, pose: Transform, name: str = ..., group: int = ...,
                     filter: int = ..., segmentationId: int = ...) -> int: ...
    ''' create_actor(self: Gym, env: Env, asset: Asset, pose: Transform, name: str = None, group: int = - 1, filter: int = - 1, segmentationId: int = 0) -> int
    Creates an Actor from an Asset
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Asset) - Asset Handle
    param3 (isaacgym.gymapi.Transform) - transform transform of where the actor will be initially placed
    param4 (str) - name of the actor
    param5 (int) - collision group that actor will be part of. The actor will not collide with anything outside of the same collisionGroup
    param6 (int) - bitwise filter for elements in the same collisionGroup to mask off collision
    param7 (int) - segmentation ID used in segmentation camera sensors
    
    
    Returns:
    Handle to actor
    
    Return type:
    Handle
    '''


    def create_aggregate(self, arg0: Env, arg1: List[int]) -> bool: ...
    ''' create_aggregate(self: Gym, arg0: Env, arg1: List[int]) -> bool
    Creates aggregate group for actors with CPU pipeline
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (list of Actor) - Actor Handles
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def create_asset_force_sensor(self, *args, **kwargs) -> Any: ...
    ''' create_asset_force_sensor(self: Gym, asset: Asset, body_idx: int, local_pose: Transform, props: ForceSensorProperties = None) -> int
    
    Creates a force sensor on a body at the provided position
    Args:asset (Asset): Asset
    body_idx (int): Rigid body index
    local_pose (isaacgym.gymapi.Transform): Local pose of the sensor relative to the body
    props (isaacgym.gymapi.ForceSensorProperties): Force sensor properties (optional)
    Returns:
    Force sensor index or -1 on failure.
    '''


    def create_box(self, sim: Sim, width: float, height: float, depth: float, options: AssetOptions = ...) -> Asset: ...
    ''' create_box(self: Gym, sim: Sim, width: float, height: float, depth: float, options: AssetOptions = None) -> Asset
    Creates a box Asset
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (float) - width of the box (x-axis)
    param3 (float) - height of the box (y-axis)
    param4 (float) - depth of the box (z-axis)
    param5 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def create_camera_sensor(self, arg0: Env, arg1: CameraProperties) -> int: ...
    ''' create_camera_sensor(self: Gym, arg0: Env, arg1: CameraProperties) -> int
    Creates Camera Sensor on given environment. The properties of the camera sensor are given by camProps. See isaacgym.gymapi.CameraProperties.
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (isaacgym.gymapi.CameraProperties) - properties of the camera sensor
    
    
    Returns:
    Camera Handle
    
    Return type:
    Handle
    '''


    def create_capsule(self, sim: Sim, radius: float, width: float, options: AssetOptions = ...) -> Asset: ...
    ''' create_capsule(self: Gym, sim: Sim, radius: float, width: float, options: AssetOptions = None) -> Asset
    Creates a Capsule mesh that extends along the x-axis with its local origin at the center of the capsule
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (float) - capsule radius
    param3 (float) - width of the capsule (x-axis)
    param4 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def create_cloth_grid(self, arg0: Sim, arg1: int, arg2: int, arg3: float, arg4: float) -> Asset: ...
    ''' create_cloth_grid(self: Gym, arg0: Sim, arg1: int, arg2: int, arg3: float, arg4: float) -> Asset
    Creates a cloth grid made of particles connected with constraints.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (int) - width of the cloth (x-axis)
    param3 (int) - length of the cloth (y-axis)
    param4 (float) - distance between particles width-wise (x-axis)
    param5 (float) - distance between particles length-wise (y-axis)
    param6 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def create_env(self, arg0: Sim, arg1: Vec3, arg2: Vec3, arg3: int) -> Env: ...
    ''' create_env(self: Gym, arg0: Sim, arg1: Vec3, arg2: Vec3, arg3: int) -> Env
    Creates one simulation Environment
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (isaacgym.gymapi.Vec3) - lower bounds of environment space
    param3 (isaacgym.gymapi.Vec3) - upper bounds of environment space
    param4 (int) - Number of environments to tile in a row
    '''


    def create_performance_timers(self, arg0: Sim) -> int: ...
    ''' create_performance_timers(self: Gym, arg0: Sim) -> int
    Creates a set of performance timers that can be queried by the user
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    Simulation Timers Handle
    
    Return type:
    Handle
    '''


    def create_rigid_body_attractor(self, arg0: Env, arg1: AttractorProperties) -> int: ...
    ''' create_rigid_body_attractor(self: Gym, arg0: Env, arg1: AttractorProperties) -> int
    Creates an attractor for the selected environment using the properties defined.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (isaacgym.gymapi.AttractorProperties) - the new properties for the attractor.
    
    
    Returns:
    Attractor Handle
    
    Return type:
    Handle
    '''


    def create_sim(self, compute_device: int = ..., graphics_device: int = ..., type: SimType = ...,
                   params: SimParams = ...) -> Sim: ...
    ''' create_sim(self: Gym, compute_device: int = 0, graphics_device: int = 0, type: SimType = SimType.SIM_FLEX, params: SimParams = <SimParams object at 0x7fb9865855e0>) -> Sim
    Allocates which device will simulate and which device will render the scene. Defines the simulation type to be used.
    Parameters:
    
    param1 (int) - index of CUDA-enabled GPU to be used for simulation.
    param2 (int) - index of GPU to be used for rendering.
    param3 (isaacgym.gymapi.SimType) - Type of simulation to be used.
    param4 (isaacgym.gymapi.SimParams) - Simulation parameters.
    
    
    Returns:
    Simulation Handle
    
    Return type:
    Sim
    '''


    def create_sphere(self, sim: Sim, radius: float, options: AssetOptions = ...) -> Asset: ...
    ''' create_sphere(self: Gym, sim: Sim, radius: float, options: AssetOptions = None) -> Asset
    Creates a sphere Asset
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (float) - sphere radius
    param3 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    @overload
    def create_tet_grid(self, arg0: Sim, arg1: SoftMaterial, arg2: int, arg3: int, arg4: int, arg5: float, arg6: float,
                        arg7: float, arg8: float, arg9: bool, arg10: bool, arg11: bool, arg12: bool) -> Asset: ...
    ''' create_tet_grid(*args, **kwargs)
    Overloaded function.
    
    create_tet_grid(self: Gym, arg0: Sim, arg1: SoftMaterial, arg2: int, arg3: int, arg4: int, arg5: float, arg6: float, arg7: float, arg8: float, arg9: bool, arg10: bool, arg11: bool, arg12: bool) -> Asset
    
    Creates a tetrahedral Grid
    Args:param1 (Sim): Simulation Handle.
    param2 (isaacgym.gymapi.SoftMaterial): soft material definitions
    param3 (int):  dimx number of tetrahedrons on x-axis
    param4 (int):  dimy number of tetrahedrons on y-axis
    param5 (int):  dimz number of tetrahedrons on z-axis
    param6 (float): spacingx length of tetrahedrons on x-axis
    param7 (float): spacingy length of tetrahedrons on y-axis
    param8 (float): spacingz length of tetrahedrons on z-axis
    param9 (float): density of tetrahedrons
    param10 (bool): when true, fixes the base of the box where it is placed.
    param11 (bool): when true, fixes the top of the box where it is placed.
    param12 (bool): when true, fixes the left side of the box where it is placed.
    param13 (bool): when true, fixes the right side of the box where it is placed.
    param14 (isaacgym.gymapi.AssetOptions): asset Options.
    
    Returns:Handle: Handle to asset

    Args:param1 (Sim): Simulation Handle.
    param2 (isaacgym.gymapi.SoftMaterial): soft material definitions
    param3 (int):  dimx number of tetrahedrons on x-axis
    param4 (int):  dimy number of tetrahedrons on y-axis
    param5 (int):  dimz number of tetrahedrons on z-axis
    param6 (float): spacingx length of tetrahedrons on x-axis
    param7 (float): spacingy length of tetrahedrons on y-axis
    param8 (float): spacingz length of tetrahedrons on z-axis
    param9 (float): density of tetrahedrons
    param10 (bool): when true, fixes the base of the box where it is placed.
    param11 (bool): when true, fixes the top of the box where it is placed.
    param12 (bool): when true, fixes the left side of the box where it is placed.
    param13 (bool): when true, fixes the right side of the box where it is placed.
    param14 (isaacgym.gymapi.AssetOptions): asset Options.
    
    Returns:Handle: Handle to asset
    '''


    @overload
    def create_tet_grid(self, arg0: Sim, arg1: SoftMaterial, arg2: int, arg3: int, arg4: int, arg5: float, arg6: float,
                        arg7: float, arg8: float, arg9: bool, arg10: bool, arg11: bool, arg12: bool) -> Asset: ...

    def create_texture_from_buffer(self, arg0: Sim, arg1: int, arg2: int, arg3: numpy.ndarray[uint8]) -> int: ...
    ''' create_texture_from_buffer(self: Gym, arg0: Sim, arg1: int, arg2: int, arg3: numpy.ndarray[uint8]) -> int
    Loads a texture from an image buffer.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (int) - Width of the texture
    param3 (int) - Height of the texture
    param4 (py::array_t) - Texture Buffer
    
    
    Returns:
    Handle to the texture or -1 if failure
    
    Return type:
    int
    '''


    def create_texture_from_file(self, arg0: Sim, arg1: str) -> int: ...
    ''' create_texture_from_file(self: Gym, arg0: Sim, arg1: str) -> int
    Loads a texture from a file.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (str) - filename of the image file containing texture
    
    
    Returns:
    Handle to the texture
    
    Return type:
    Handle
    '''


    def create_usd_exporter(self, options: UsdExportOptions = ...) -> UsdExporter: ...
    ''' create_usd_exporter(self: Gym, options: UsdExportOptions = None) -> UsdExporter
    Destroy USD exporter
    Parameters:
    param1 (isaacgym.gymapi.UsdExportOptions) - USD Exporter Options
    
    Returns:
    Handle to Exporter
    
    Return type:
    Handle
    '''


    def create_viewer(self, arg0: Sim, arg1: CameraProperties) -> Viewer: ...
    ''' create_viewer(self: Gym, arg0: Sim, arg1: CameraProperties) -> Viewer
    Creates a viewer for the simulation.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (isaacgym.gymapi.CameraProperties) - Camera properties for default camera in viewer
    '''


    def debug_print_asset(self, asset: Asset, outpath: str = ...) -> None: ...
    ''' debug_print_asset(self: Gym, asset: Asset, outpath: str = None) -> None
    Outputs assets properties of asset to std out or to file.
    Parameters:
    
    param1 (isaacgym.gymapi.Asset) - Asset
    param2 (string) - Output file path
    '''


    def destroy_camera_sensor(self, arg0: Sim, arg1: Env, arg2: int) -> None: ...
    ''' destroy_camera_sensor(self: Gym, arg0: Sim, arg1: Env, arg2: int) -> None
    Destroys all data referring to given camera sensor
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Camera) - Handle to the camera
    '''


    def destroy_env(self, arg0: Env) -> None: ...
    ''' destroy_env(self: Gym, arg0: Env) -> None
    Deletes an environment and releases memory used from it
    Parameters:
    param1 (Env) - Environment Handle.
    '''


    def destroy_performance_timers(self, arg0: Sim, arg1: int) -> None: ...
    ''' destroy_performance_timers(self: Gym, arg0: Sim, arg1: int) -> None
    Destroys internal timers
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (Timers) - Simulation Timers Handle
    '''


    def destroy_sim(self, arg0: Sim) -> None: ...
    ''' destroy_sim(self: Gym, arg0: Sim) -> None
    Cleans up all remaining handles to the simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def destroy_usd_exporter(self, arg0: UsdExporter) -> None: ...
    ''' destroy_usd_exporter(self: Gym, arg0: UsdExporter) -> None
    Destroy USD exporter
    Parameters:
    param1 (Exporter) - USD Exporter Handle
    '''


    def destroy_viewer(self, arg0: Viewer) -> None: ...
    ''' destroy_viewer(self: Gym, arg0: Viewer) -> None
    Closes viewer and destroys its handle
    Parameters:
    param1 (Viewer) - Viewer Handle.
    '''


    def draw_env_rigid_contacts(self, arg0: Viewer, arg1: Env, arg2: Vec3, arg3: float, arg4: bool) -> None: ...
    ''' draw_env_rigid_contacts(self: Gym, arg0: Viewer, arg1: Env, arg2: Vec3, arg3: float, arg4: bool) -> None
    Draw Contact Forces for all Rigid Bodies in an Env.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (isaacgym.gymapi.Vec3) - contact's color.
    param3 (float) - scale to draw contact force vectors.
    param4 (float) - apply square root scale when drawing contact force vectors
    '''


    def draw_env_soft_contacts(self, arg0: Viewer, arg1: Env, arg2: Vec3, arg3: float, arg4: bool,
                               arg5: bool) -> None: ...
    ''' draw_env_soft_contacts(self: Gym, arg0: Viewer, arg1: Env, arg2: Vec3, arg3: float, arg4: bool, arg5: bool) -> None
    Draw Contact Forces for all soft Bodies in an Env.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (isaacgym.gymapi.Vec3) - contact's color.
    param3 (float) - scale to draw contact force vectors.
    param4 (bool) - apply square root scale when drawing contact force vectors
    param5 (bool) - force visualization
    '''


    def draw_viewer(self, viewer: Viewer, sim: Sim, render_collision: bool = ...) -> None: ...
    ''' draw_viewer(self: Gym, viewer: Viewer, sim: Sim, render_collision: bool = True) -> None
    Renders the viewer
    Parameters:
    
    param1 (Viewer) - Viewer Handle.
    param2 (Sim) - Simulation Handle.
    param3 (bool) - renterCollisionMeshes flag to determine if should render the collision meshes instead of display meshes
    '''


    def enable_actor_dof_force_sensors(self, arg0: Env, arg1: int) -> bool: ...
    ''' enable_actor_dof_force_sensors(self: Gym, arg0: Env, arg1: int) -> bool
    Enables DOF force collection for the actor's degrees of freedom.
    Parameters:
    
    ( (param1) - obj: list of Env): Environment Handles
    param2 (Actor) - Actor Handle
    
    
    Returns:
    True if DOF force collection is supported for this actor, False otherwise.
    '''


    def end_access_image_tensors(self, arg0: Sim) -> None: ...
    ''' end_access_image_tensors(self: Gym, arg0: Sim) -> None
    Terminates access to image tensors. Releases data from all image tensors to the GPU
    Parameters:
    param1 (Sim) - Simulation Handle.
    '''


    def end_aggregate(self, arg0: Env) -> bool: ...
    ''' end_aggregate(self: Gym, arg0: Env) -> bool
    Ends current aggregate group
    Parameters:
    param1 (Env) - Environment Handle
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def export_usd_asset(self, arg0: UsdExporter, arg1: Asset, arg2: str) -> bool: ...
    ''' export_usd_asset(self: Gym, arg0: UsdExporter, arg1: Asset, arg2: str) -> bool
    Exports asset in USD Format
    Parameters:
    
    param1 (Exporter) - USD Exporter Handle
    param2 (Asset) - Asset Handle
    param3 (str) - path and file name to save the asset.
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def export_usd_sim(self, exporter: UsdExporter, sim: Sim, name: str, dirName: str = ...) -> object: ...
    ''' export_usd_sim(self: Gym, exporter: UsdExporter, sim: Sim, name: str, dirName: str = None) -> object
    Exports simulation in USD Format
    Parameters:
    
    param1 (isaacgym.gymapi.UsdExporter) - USD Exporter Handle
    param2 (Sim) - Simulation Handle
    param3 (str) - path and file name to save the asset.
    
    
    Returns:
    obj: map to transforms : returns none if failed
    '''


    def fetch_results(self, arg0: Sim, arg1: bool) -> None: ...
    ''' fetch_results(self: Gym, arg0: Sim, arg1: bool) -> None
    Populates Host buffers for the simulation from Device values
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (bool) - flags if should wait for latest simulation step to complete.
    '''


    def find_actor_actuator_index(self, arg0: Env, arg1: int, arg2: str) -> int: ...
    ''' find_actor_actuator_index(self: Gym, arg0: Env, arg1: int, arg2: str) -> int
    Gets the index of a named actuator of the actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (str) - Name of the actuator
    
    
    Returns:
    index of the actuator
    
    Return type:
    int
    '''


    def find_actor_dof_handle(self, arg0: Env, arg1: int, arg2: str) -> int: ...
    ''' find_actor_dof_handle(self: Gym, arg0: Env, arg1: int, arg2: str) -> int
    Finds actor Degree of Freedom handle given its name
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (str) - name of DOF
    
    
    Returns:
    DOF Handle
    
    Return type:
    Handle
    '''


    def find_actor_dof_index(self, arg0: Env, arg1: int, arg2: str, arg3: IndexDomain) -> int: ...
    ''' find_actor_dof_index(self: Gym, arg0: Env, arg1: int, arg2: str, arg3: IndexDomain) -> int
    Find the array index of a named degree-of-freedom.
    Use domain eActorDomain to get an index into arrays returned by functions like
    isaacgym.gymapi.Gym.get_actor_dof_states or isaacgym.gymapi.Gym.get_actor_dof_properties.
    Currently, the other domains are not useful, because there is no API for dealing with DOFs at the env or sim level.
    This may change in the future.
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (str) - Name of the DOF
    param4 (isaacgym.gymapi.IndexDomain) - determines which state buffer to get the index for (env or sim).
    
    
    Returns:
    index of the DOF in the specified domain
    
    Return type:
    int
    '''


    def find_actor_fixed_tendon_joint_index(self, arg0: Env, arg1: int, arg2: int, arg3: str) -> int: ...
    ''' find_actor_fixed_tendon_joint_index(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: str) -> int
    Gets the name of a joint in a fixed tendon of an actor at the indexes provided
    
    Note
    The order of joints in an actor tendon may differ from the order in the asset.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the tendon
    param4 (str) - Name of the joint
    
    
    Returns:
    index of the joint, or -1 if not found
    
    Return type:
    int
    '''


    def find_actor_handle(self, arg0: Env, arg1: str) -> int: ...
    ''' find_actor_handle(self: Gym, arg0: Env, arg1: str) -> int
    Gets handle for an actor, given its name
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (str) - name of actor
    
    
    Returns:
    Handle to actor
    
    Return type:
    Handle
    '''


    def find_actor_index(self, arg0: Env, arg1: str, arg2: IndexDomain) -> int: ...
    ''' find_actor_index(self: Gym, arg0: Env, arg1: str, arg2: IndexDomain) -> int
    Gets index of actor in domain from actor name
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (str) - Actor Name
    param3 (isaacgym.gymapi.IndexDomain) - Simulation, Environment, or Actor domain
    
    
    Returns:
    Actor Index
    
    Return type:
    int
    '''


    def find_actor_joint_handle(self, arg0: Env, arg1: int, arg2: str) -> int: ...
    ''' find_actor_joint_handle(self: Gym, arg0: Env, arg1: int, arg2: str) -> int
    Finds actor Joint handle given its name
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (str) - joint name
    
    
    Returns:
    Joint Handle
    
    Return type:
    Handle
    '''


    def find_actor_joint_index(self, arg0: Env, arg1: int, arg2: str, arg3: IndexDomain) -> int: ...
    ''' find_actor_joint_index(self: Gym, arg0: Env, arg1: int, arg2: str, arg3: IndexDomain) -> int
    Use this function to find the array index of a named joint.
    Currently, this function is not useful, because there is no API for dealing with joint arrays.  This may change in the future.
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (str) - name of the joint.
    param4 (isaacgym.gymapi.IndexDomain) - determines which state buffer to get the index for (env or sim).
    
    
    Returns:
    index of the joint in the specified domain
    
    Return type:
    Handle
    '''


    def find_actor_rigid_body_handle(self, arg0: Env, arg1: int, arg2: str) -> int: ...
    ''' find_actor_rigid_body_handle(self: Gym, arg0: Env, arg1: int, arg2: str) -> int
    Finds actor rigid body handle given its name
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (str) - name of the rigid body.
    
    
    Returns:
    Rigid Body Handle
    
    Return type:
    Handle
    '''


    def find_actor_rigid_body_index(self, arg0: Env, arg1: int, arg2: str, arg3: IndexDomain) -> int: ...
    ''' find_actor_rigid_body_index(self: Gym, arg0: Env, arg1: int, arg2: str, arg3: IndexDomain) -> int
    Use this function to find the index of a rigid body in a state buffer.
    
    Use domain DOMAIN_ENV to get an index into the state buffer returned by isaacgym.gymapi.Gym.get_env_rigid_body_states.
    Use domain DOMAIN_SIM to get an index into the state buffer returned by isaacgym.gymapi.Gym.get_sim_rigid_body_states.
    Use domain DOMAIN_ACTOR to get an index into the state buffer returned by isaacgym.gymapi.Gym.get_actor_rigid_body_states
    Use domain DOMAIN_ACTOR to get an index into the property buffer returned by isaacgym.gymapi.Gym.get_actor_rigid_body_properties.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (str) - name of the rigid body.
    param4 (isaacgym.gymapi.IndexDomain) - determines which state buffer to get the index for (env or sim).
    
    
    Returns:
    index of the rigid body in the specified domain
    
    Return type:
    int
    '''


    def find_actor_tendon_index(self, arg0: Env, arg1: int, arg2: str) -> int: ...
    ''' find_actor_tendon_index(self: Gym, arg0: Env, arg1: int, arg2: str) -> int
    Gets the index of a named tendon of the actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (str) - Name of the tendon
    
    
    Returns:
    index of the tendon
    
    Return type:
    int
    '''


    def find_asset_actuator_index(self, arg0: Asset, arg1: str) -> int: ...
    ''' find_asset_actuator_index(self: Gym, arg0: Asset, arg1: str) -> int
    Gets the index of a named actuator in the asset's actuator array
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (str) - name of the actuator
    
    
    Returns:
    index of the actuator
    
    Return type:
    int
    '''


    def find_asset_dof_index(self, arg0: Asset, arg1: str) -> int: ...
    ''' find_asset_dof_index(self: Gym, arg0: Asset, arg1: str) -> int
    Gets the index of a named degree-of-freedom in the asset's DOF array
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (str) - name of the DOF
    
    
    Returns:
    index of the DOF
    
    Return type:
    int
    '''


    def find_asset_joint_index(self, arg0: Asset, arg1: str) -> int: ...
    ''' find_asset_joint_index(self: Gym, arg0: Asset, arg1: str) -> int
    Gets the index of a named joint in the asset's joint array
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (str) - name of the joint
    
    
    Returns:
    index of the joint
    
    Return type:
    int
    '''


    def find_asset_rigid_body_index(self, arg0: Asset, arg1: str) -> int: ...
    ''' find_asset_rigid_body_index(self: Gym, arg0: Asset, arg1: str) -> int
    Gets the index of a named rigid body in the asset's body array
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (str) - name of the rigid body
    
    
    Returns:
    index of the rigid body
    
    Return type:
    int
    '''


    def find_asset_tendon_index(self, arg0: Asset, arg1: str) -> int: ...
    ''' find_asset_tendon_index(self: Gym, arg0: Asset, arg1: str) -> int
    Gets the index of a named tendon in the asset's tendon array
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (str) - name of the tendon
    
    
    Returns:
    index of the tendon
    
    Return type:
    int
    '''


    def free_texture(self, arg0: Sim, arg1: int) -> None: ...
    ''' free_texture(self: Gym, arg0: Sim, arg1: int) -> None
    releases texture data from memory
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Texture) - Texture handle to free.
    '''


    def get_actor_actuator_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_actuator_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of actuators for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    number of actuators in actor
    
    Return type:
    int
    '''


    def get_actor_actuator_joint_name(self, arg0: Env, arg1: int, arg2: int) -> str: ...
    ''' get_actor_actuator_joint_name(self: Gym, arg0: Env, arg1: int, arg2: int) -> str
    Gets the name of an actuator for an actor at the index provided
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the actuator
    
    
    Returns:
    name of actuator in asset
    
    Return type:
    str
    '''


    def get_actor_actuator_name(self, arg0: Env, arg1: int, arg2: int) -> str: ...
    ''' get_actor_actuator_name(self: Gym, arg0: Env, arg1: int, arg2: int) -> str
    Gets the name of an actuator for an actor at the index provided
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the actuator
    
    
    Returns:
    name of actuator in asset
    
    Return type:
    str
    '''


    def get_actor_actuator_properties(self, arg0: Env, arg1: int) -> List[ActuatorProperties]: ...
    ''' get_actor_actuator_properties(self: Gym, arg0: Env, arg1: int) -> List[ActuatorProperties]
    Gets an array of actuator properties for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    array containing isaacgym.gymapi.ActuatorProperties structure
    
    Return type:
    array of isaacgym.gymapi.ActuatorProperties
    '''


    def get_actor_asset(self, arg0: Env, arg1: int) -> Asset: ...
    ''' get_actor_asset(self: Gym, arg0: Env, arg1: int) -> Asset
    Gets Asset for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    Asset Handle
    
    Return type:
    Handle
    '''


    def get_actor_count(self, arg0: Env) -> int: ...
    ''' get_actor_count(self: Gym, arg0: Env) -> int
    Gets number of actors in an environment
    Parameters:
    param1 (Env) - Environment Handle.
    
    Returns:
    name of actors in environment
    
    Return type:
    int
    '''


    def get_actor_dof_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_dof_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of Degree of Freedom for an actor
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    
    
    Returns:
    number of DOFs in actor
    
    Return type:
    int
    '''


    def get_actor_dof_dict(self, arg0: Env, arg1: int) -> Dict[str, int]: ...
    ''' get_actor_dof_dict(self: Gym, arg0: Env, arg1: int) -> Dict[str, int]
    maps degree of freedom names to actor-relative indices
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: dict of str : dictionary of actor degree of freedom names
    '''


    def get_actor_dof_forces(self, arg0: Env, arg1: int) -> numpy.ndarray[numpy.float32]: ...
    ''' get_actor_dof_forces(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[float32]
    Gets forces for the actor's degrees of freedom.
    If the Degree of Freedom is linear, the force is a force in Newtons.
    If the Degree of Freedom is revolute, the force is a torque in Nm.
    Parameters:
    
    ( (param1) - obj: list of Env): Environment Handles
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of float : array of DOF forces
    '''


    def get_actor_dof_frames(self, arg0: Env, arg1: int) -> numpy.ndarray[DofFrame]: ...
    ''' get_actor_dof_frames(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[DofFrame]
    Gets Frames for Degrees of Freedom of actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of isaacgym.gymapi.DofFrame : List of actor dof frames
    '''


    def get_actor_dof_handle(self, arg0: Env, arg1: int, arg2: int) -> int: ...
    ''' get_actor_dof_handle(self: Gym, arg0: Env, arg1: int, arg2: int) -> int
    Gets number of Degree of Freedom for an actor
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (int) - index of DOF
    
    
    Returns:
    DOF Handle
    
    Return type:
    Handle
    '''


    def get_actor_dof_index(self, arg0: Env, arg1: int, arg2: int, arg3: IndexDomain) -> int: ...
    ''' get_actor_dof_index(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: IndexDomain) -> int
    Currently, this function is not useful, because there is no API for dealing with DOF arrays at the env or sim level.
    This may change in the future.
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (int) - Index of the DOF
    param4 (isaacgym.gymapi.IndexDomain) - determines which state buffer to get the index for (env or sim).
    
    
    Returns:
    Handle of the DOF in the specified domain
    
    Return type:
    Handle
    '''


    def get_actor_dof_names(self, arg0: Env, arg1: int) -> List[str]: ...
    ''' get_actor_dof_names(self: Gym, arg0: Env, arg1: int) -> List[str]
    Gets names of all degrees of freedom on actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of str : list of actor degree of freedon names
    '''


    def get_actor_dof_position_targets(self, *args, **kwargs) -> Any: ...
    ''' get_actor_dof_position_targets(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[float32]
    
    Gets target position for the actor's degrees of freedom. if the joint is prismatic, the target is in meters. if the joint is revolute, the target is in radians.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of float : array of position targets
    '''


    def get_actor_dof_properties(self, *args, **kwargs) -> Any: ...
    ''' get_actor_dof_properties(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[carb::gym::GymDofProperties]
    Gets properties for all Dofs on an actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    A structured Numpy array of DOF properties.
    '''


    def get_actor_dof_states(self, *args, **kwargs) -> Any: ...
    ''' get_actor_dof_states(self: Gym, arg0: Env, arg1: int, arg2: int) -> numpy.ndarray[DofState]
    
    Gets state for the actor's degrees of freedom. see isaacgym.gymapi.DofState
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - flags for the state to update, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    obj: list of isaacgym.gymapi.DofState : array of states
    '''


    def get_actor_dof_velocity_targets(self, *args, **kwargs) -> Any: ...
    ''' get_actor_dof_velocity_targets(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[float32]
    
    Gets target velocities for the actor's degrees of freedom. if the joint is prismatic, the target is in m/s. if the joint is revolute, the target is in rad/s.
    Parameters:
    
    ( (param1) - obj: list of Env): Environment Handles
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of float : array of velocity targets
    '''


    def get_actor_fixed_tendon_joint_coefficients(self, arg0: Env, arg1: int, arg2: int) -> List[float]: ...
    ''' get_actor_fixed_tendon_joint_coefficients(self: Gym, arg0: Env, arg1: int, arg2: int) -> List[float]
    Gets an array of tendon joint coefficients for the given actor and tendon index
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - index of tendon to get joint coefficients of
    
    
    Returns:
    array containing float coefficients
    
    Return type:
    array of float
    '''


    def get_actor_fixed_tendon_joint_name(self, arg0: Env, arg1: int, arg2: int, arg3: int) -> str: ...
    ''' get_actor_fixed_tendon_joint_name(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: int) -> str
    Gets the name of a joint in a fixed tendon of an actor at the indexes provided.
    
    Note
    The order of joints in an actor tendon may differ from the order in the asset.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the tendon
    param4 (int) - Index of the joint
    
    
    Returns:
    name of joint in fixed tendon of actor
    
    Return type:
    str
    '''


    def get_actor_force_sensor(self, arg0: Env, arg1: int, arg2: int) -> ForceSensor: ...
    ''' get_actor_force_sensor(self: Gym, arg0: Env, arg1: int, arg2: int) -> ForceSensor
    Gets a force sensor instance for the given actor
    Parameters:
    
    param1 (Env) - Environment
    param2 (Handle) - Actor handle
    param3 (int) - Force sensor index
    
    
    Returns:
    the force sensor instance or None on failure.
    
    Return type:
    isaacgym.gymapi.ForceSensor
    '''


    def get_actor_force_sensor_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_force_sensor_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of force sensors in the specified actor
    Parameters:
    
    param1 (Env) - Environment
    param2 (Handle) - Actor handle
    
    
    Returns:
    Number of sensors
    
    Return type:
    int
    '''


    def get_actor_handle(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_handle(self: Gym, arg0: Env, arg1: int) -> int
    Gets handle for an actor, given its index
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (int) - actor index
    
    
    Returns:
    Handle to actor
    
    Return type:
    Handle
    '''


    def get_actor_index(self, arg0: Env, arg1: int, arg2: IndexDomain) -> int: ...
    ''' get_actor_index(self: Gym, arg0: Env, arg1: int, arg2: IndexDomain) -> int
    Gets index of actor in domain from actor handle
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (isaacgym.gymapi.IndexDomain) - Simulation, Environment, or Actor domain
    
    
    Returns:
    Actor Index
    
    Return type:
    int
    '''


    def get_actor_joint_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_joint_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of joints for an actor
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    
    
    Returns:
    number of joints in actor
    
    Return type:
    int
    '''


    def get_actor_joint_dict(self, arg0: Env, arg1: int) -> Dict[str, int]: ...
    ''' get_actor_joint_dict(self: Gym, arg0: Env, arg1: int) -> Dict[str, int]
    maps joint names to actor-relative indices
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: dict of str : dictionary of actor joint names
    '''


    def get_actor_joint_handle(self, arg0: Env, arg1: int, arg2: int) -> int: ...
    ''' get_actor_joint_handle(self: Gym, arg0: Env, arg1: int, arg2: int) -> int
    Gets actor joint handle given its index
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (int) - index of joint
    
    
    Returns:
    Joint Handle
    
    Return type:
    Handle
    '''


    def get_actor_joint_index(self, arg0: Env, arg1: int, arg2: int, arg3: IndexDomain) -> int: ...
    ''' get_actor_joint_index(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: IndexDomain) -> int
    Get the array index of a named joint
    Currently, this function is not useful, because there is no API for dealing with joint arrays.  This may change in the future.
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle.
    param3 (int) - index of joint
    param4 (isaacgym.gymapi.IndexDomain) - determines which state buffer to get the index for (env or sim).
    
    
    Returns:
    index of the joint in the specified domain
    
    Return type:
    Handle
    '''


    def get_actor_joint_names(self, arg0: Env, arg1: int) -> List[str]: ...
    ''' get_actor_joint_names(self: Gym, arg0: Env, arg1: int) -> List[str]
    Gets names of all joints on actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of str : list of actor joint names
    '''


    def get_actor_joint_transforms(self, arg0: Env, arg1: int) -> numpy.ndarray[Transform]: ...
    ''' get_actor_joint_transforms(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[Transform]
    Gets Transforms for Joints on actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of isaacgym.gymapi.Transform : List of actor joint transforms
    '''


    def get_actor_name(self, arg0: Env, arg1: int) -> str: ...
    ''' get_actor_name(self: Gym, arg0: Env, arg1: int) -> str
    Gets name of actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    Actor Name
    
    Return type:
    str
    '''


    def get_actor_rigid_body_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_rigid_body_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of rigid bodies for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    number of rigid bodies in actor
    
    Return type:
    int
    '''


    def get_actor_rigid_body_dict(self, arg0: Env, arg1: int) -> Dict[str, int]: ...
    ''' get_actor_rigid_body_dict(self: Gym, arg0: Env, arg1: int) -> Dict[str, int]
    maps rigid body names to actor-relative indices
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: dict of str : dictionary of actor rigid body names
    '''


    def get_actor_rigid_body_handle(self, arg0: Env, arg1: int, arg2: int) -> int: ...
    ''' get_actor_rigid_body_handle(self: Gym, arg0: Env, arg1: int, arg2: int) -> int
    Gets actor rigid body handle given its index
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Rigid Body Index
    
    
    Returns:
    Rigid Body Handle
    
    Return type:
    Handle
    '''


    def get_actor_rigid_body_index(self, arg0: Env, arg1: int, arg2: int, arg3: IndexDomain) -> int: ...
    ''' get_actor_rigid_body_index(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: IndexDomain) -> int
    Use this function to get the index of a rigid body in a state buffer.
    
    Use domain DOMAIN_ENV to get an index into the state buffer returned by isaacgym.gymapi.Gym.get_env_rigid_body_states.
    Use domain DOMAIN_SIM to get an index into the state buffer returned by isaacgym.gymapi.Gym.get_sim_rigid_body_states.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - index of the rigid body in the actor rigid body array.
    param4 (isaacgym.gymapi.IndexDomain) - determines which state buffer to get the index for (env or sim).
    
    
    Returns:
    index of the rigid body in the specified domain
    
    Return type:
    int
    '''


    def get_actor_rigid_body_names(self, arg0: Env, arg1: int) -> List[str]: ...
    ''' get_actor_rigid_body_names(self: Gym, arg0: Env, arg1: int) -> List[str]
    Gets names of all rigid bodies on actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of str : list of actor rigid body names
    '''


    def get_actor_rigid_body_properties(self, arg0: Env, arg1: int) -> List[RigidBodyProperties]: ...
    ''' get_actor_rigid_body_properties(self: Gym, arg0: Env, arg1: int) -> List[RigidBodyProperties]
    Gets properties for rigid bodies in an actor on selected environment.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of isaacgym.gymapi.RigidBodyProperties : list of rigid body properties
    '''


    def get_actor_rigid_body_shape_indices(self, arg0: Env, arg1: int) -> List[IndexRange]: ...
    ''' get_actor_rigid_body_shape_indices(self: Gym, arg0: Env, arg1: int) -> List[IndexRange]
    Maps actor body shapes to index ranges in shape array
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of isaacgym.gymapi.IndexRange : indices
    '''


    def get_actor_rigid_body_states(self, arg0: Env, arg1: int, arg2: int) -> numpy.ndarray[RigidBodyState]: ...
    ''' get_actor_rigid_body_states(self: Gym, arg0: Env, arg1: int, arg2: int) -> numpy.ndarray[RigidBodyState]
    Gets state for the actors's Rigid Bodies. see isaacgym.gymapi.RigidBodyState.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Actor) - Actor Handle
    param2 (int) - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    obj: list of isaacgym.gymapi.RigidBodyState : List of rigid body states
    '''


    def get_actor_rigid_shape_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_rigid_shape_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets count of actor shapes
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    number of shapes in actor
    
    Return type:
    int
    '''


    def get_actor_rigid_shape_properties(self, arg0: Env, arg1: int) -> List[RigidShapeProperties]: ...
    ''' get_actor_rigid_shape_properties(self: Gym, arg0: Env, arg1: int) -> List[RigidShapeProperties]
    Gets properties for rigid shapes in an actor on selected environment.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of isaacgym.gymapi.RigidShapeProperties : list of rigid shape properties
    '''


    def get_actor_root_rigid_body_handle(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_root_rigid_body_handle(self: Gym, arg0: Env, arg1: int) -> int
    Get the handle of the root rigid body of an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    Root Rigid Body Handle
    
    Return type:
    Handle
    '''


    def get_actor_scale(self, arg0: Env, arg1: int) -> float: ...
    ''' get_actor_scale(self: Gym, arg0: Env, arg1: int) -> float
    Gets the scale of the actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    The current scale factor of the actor.
    '''


    def get_actor_soft_body_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_soft_body_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of soft bodies for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    number of soft bodies in actor
    
    Return type:
    int
    '''


    def get_actor_soft_materials(self, arg0: Env, arg1: int) -> List[SoftMaterial]: ...
    ''' get_actor_soft_materials(self: Gym, arg0: Env, arg1: int) -> List[SoftMaterial]
    Gets properties for all soft materials on an actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    obj: list of isaacgym.gymapi.SoftMaterials: Array of soft materials
    '''


    def get_actor_tendon_count(self, arg0: Env, arg1: int) -> int: ...
    ''' get_actor_tendon_count(self: Gym, arg0: Env, arg1: int) -> int
    Gets number of tendons for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    number of tendons in actor
    
    Return type:
    int
    '''


    def get_actor_tendon_name(self, arg0: Env, arg1: int, arg2: int) -> str: ...
    ''' get_actor_tendon_name(self: Gym, arg0: Env, arg1: int, arg2: int) -> str
    Gets the name of a tendon for an actor at the index provided
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the tendon
    
    
    Returns:
    name of tendon in asset
    
    Return type:
    str
    '''


    def get_actor_tendon_offset(self, arg0: Env, arg1: int, arg2: int) -> float: ...
    ''' get_actor_tendon_offset(self: Gym, arg0: Env, arg1: int, arg2: int) -> float
    Gets the length offset of a tendon of an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the tendon
    
    
    Returns:
    Tendon offset or FLOAT_MAX if invalid handles/indices provided
    
    Return type:
    float
    '''


    def get_actor_tendon_properties(self, arg0: Env, arg1: int) -> List[TendonProperties]: ...
    ''' get_actor_tendon_properties(self: Gym, arg0: Env, arg1: int) -> List[TendonProperties]
    Gets an array of tendon properties for an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    array containing isaacgym.gymapi.TendonProperties structure
    
    Return type:
    array of isaacgym.gymapi.TendonProperties
    '''


    def get_actor_tetrahedra_range(self, arg0: Env, arg1: int, arg2: int) -> IndexRange: ...
    ''' get_actor_tetrahedra_range(self: Gym, arg0: Env, arg1: int, arg2: int) -> IndexRange
    Gets the tetrahedra range for a given actor and soft body link
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of soft body in this actor
    
    
    Returns:
    Range (start, count)
    '''


    def get_actor_triangle_range(self, arg0: Env, arg1: int, arg2: int) -> IndexRange: ...
    ''' get_actor_triangle_range(self: Gym, arg0: Env, arg1: int, arg2: int) -> IndexRange
    Gets the triangle range for a given actor and soft body link
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of soft body in this actor
    
    
    Returns:
    Range (start, count)
    '''


    def get_asset_actuator_count(self, arg0: Asset) -> int: ...
    ''' get_asset_actuator_count(self: Gym, arg0: Asset) -> int
    Gets the count of actuators on a given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    number of actuators in asset
    
    Return type:
    int
    '''


    def get_asset_actuator_joint_name(self, arg0: Asset, arg1: int) -> str: ...
    ''' get_asset_actuator_joint_name(self: Gym, arg0: Asset, arg1: int) -> str
    Gets the name of an actuator of the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - Index of the actuator
    
    
    Returns:
    name of actuator in asset
    
    Return type:
    str
    '''


    def get_asset_actuator_name(self, arg0: Asset, arg1: int) -> str: ...
    ''' get_asset_actuator_name(self: Gym, arg0: Asset, arg1: int) -> str
    Gets the name of an actuator of the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - Index of the actuator
    
    
    Returns:
    name of actuator in asset
    
    Return type:
    str
    '''


    def get_asset_actuator_properties(self, arg0: Asset) -> List[ActuatorProperties]: ...
    ''' get_asset_actuator_properties(self: Gym, arg0: Asset) -> List[ActuatorProperties]
    Gets an array of actuator properties for the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    array containing isaacgym.gymapi.ActuatorProperties structure
    
    Return type:
    array of isaacgym.gymapi.ActuatorProperties
    '''


    def get_asset_dof_count(self, arg0: Asset) -> int: ...
    ''' get_asset_dof_count(self: Gym, arg0: Asset) -> int
    Gets the count of Degrees of Freedom on a given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    number of degrees of freedom in asset
    
    Return type:
    int
    '''


    def get_asset_dof_dict(self, arg0: Asset) -> Dict[str, int]: ...
    ''' get_asset_dof_dict(self: Gym, arg0: Asset) -> Dict[str, int]
    Maps dof names to asset-relative indices
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    dictionary containing mapping between dof names and asset relative indices
    
    Return type:
    dict of str
    '''


    def get_asset_dof_name(self, arg0: Asset, arg1: int) -> str: ...
    ''' get_asset_dof_name(self: Gym, arg0: Asset, arg1: int) -> str
    Gets the name of the degree of freedom on the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - index of the DOF
    
    
    Returns:
    name of DOF in asset
    
    Return type:
    str
    '''


    def get_asset_dof_names(self, arg0: Asset) -> List[str]: ...
    ''' get_asset_dof_names(self: Gym, arg0: Asset) -> List[str]
    Get list of asset DOF names
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    list of degree of freedom names in asset
    
    Return type:
    list of str
    '''


    def get_asset_dof_properties(self, *args, **kwargs) -> Any: ...
    ''' get_asset_dof_properties(self: Gym, arg0: Asset) -> numpy.ndarray[carb::gym::GymDofProperties]
    Gets an array of DOF properties for the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    A structured Numpy array of DOF properties.
    '''


    def get_asset_dof_type(self, arg0: Asset, arg1: int) -> DofType: ...
    ''' get_asset_dof_type(self: Gym, arg0: Asset, arg1: int) -> DofType
    Gets the Degree of Freedom type on the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - index of the joint
    
    
    Returns:
    type of degree of freedom
    
    Return type:
    isaacgym.gymapi.DofType
    '''


    def get_asset_fixed_tendon_joint_coefficients(self, arg0: Asset, arg1: int) -> List[float]: ...
    ''' get_asset_fixed_tendon_joint_coefficients(self: Gym, arg0: Asset, arg1: int) -> List[float]
    Gets an array of tendon joint coefficients for the given asset and tendon index
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - index of tendon to get joint coefficients of
    
    
    Returns:
    array containing float coefficients
    
    Return type:
    array of float
    '''


    def get_asset_fixed_tendon_joint_name(self, arg0: Asset, arg1: int, arg2: int) -> str: ...
    ''' get_asset_fixed_tendon_joint_name(self: Gym, arg0: Asset, arg1: int, arg2: int) -> str
    Gets the name of a joint in a fixed tendon of the given asset at the indexes provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - Index of the tendon
    param3 (int) - Index of the joint
    
    
    Returns:
    name of joint in fixed tendon in asset
    
    Return type:
    str
    '''


    def get_asset_joint_count(self, arg0: Asset) -> int: ...
    ''' get_asset_joint_count(self: Gym, arg0: Asset) -> int
    Gets the count of joints on the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    number of joints in asset
    
    Return type:
    int
    '''


    def get_asset_joint_dict(self, arg0: Asset) -> Dict[str, int]: ...
    ''' get_asset_joint_dict(self: Gym, arg0: Asset) -> Dict[str, int]
    Maps joint names to asset-relative indices
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    dictionary containing mapping between joint names and asset relative indices
    
    Return type:
    dict of str
    '''


    def get_asset_joint_name(self, arg0: Asset, arg1: int) -> str: ...
    ''' get_asset_joint_name(self: Gym, arg0: Asset, arg1: int) -> str
    Gets the name of the joint on the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - index of the joint
    
    
    Returns:
    name of joint in asset
    
    Return type:
    str
    '''


    def get_asset_joint_names(self, arg0: Asset) -> List[str]: ...
    ''' get_asset_joint_names(self: Gym, arg0: Asset) -> List[str]
    Get list of asset joint names
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    list of joint names in asset
    
    Return type:
    list of str
    '''


    def get_asset_joint_type(self, arg0: Asset, arg1: int) -> JointType: ...
    ''' get_asset_joint_type(self: Gym, arg0: Asset, arg1: int) -> JointType
    Gets the joint type on the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - index of the joint
    
    
    Returns:
    type of joint
    
    Return type:
    isaacgym.gymapi.JointType
    '''


    def get_asset_rigid_body_count(self, arg0: Asset) -> int: ...
    ''' get_asset_rigid_body_count(self: Gym, arg0: Asset) -> int
    Gets the count of rigid bodies on the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    number of rigid bodies in asset
    
    Return type:
    int
    '''


    def get_asset_rigid_body_dict(self, arg0: Asset) -> Dict[str, int]: ...
    ''' get_asset_rigid_body_dict(self: Gym, arg0: Asset) -> Dict[str, int]
    Maps rigid body names to asset-relative indices
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    dictionary containing mapping between body names and asset relative indices
    
    Return type:
    dict of str
    '''


    def get_asset_rigid_body_name(self, arg0: Asset, arg1: int) -> str: ...
    ''' get_asset_rigid_body_name(self: Gym, arg0: Asset, arg1: int) -> str
    Gets the name of the rigid body on the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - index of the rigid body
    
    
    Returns:
    name of rigid body
    
    Return type:
    str
    '''


    def get_asset_rigid_body_names(self, arg0: Asset) -> List[str]: ...
    ''' get_asset_rigid_body_names(self: Gym, arg0: Asset) -> List[str]
    Gets names of rigid bodies in asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    list rigid body names.
    '''


    def get_asset_rigid_body_shape_indices(self, arg0: Asset) -> List[IndexRange]: ...
    ''' get_asset_rigid_body_shape_indices(self: Gym, arg0: Asset) -> List[IndexRange]
    Maps asset body index to index ranges in shape array, i.e.
    the range at index i will map to the indices of all of body i's shapes
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    obj: list of isaacgym.gymapi.IndexRange : indices
    '''


    def get_asset_rigid_shape_count(self, arg0: Asset) -> int: ...
    ''' get_asset_rigid_shape_count(self: Gym, arg0: Asset) -> int
    Gets number of rigid shapes in asset
    Parameters:
    param1 (Asset) - Asset
    
    Returns:
    number of rigid shapes
    
    Return type:
    int
    '''


    def get_asset_rigid_shape_properties(self, arg0: Asset) -> List[RigidShapeProperties]: ...
    ''' get_asset_rigid_shape_properties(self: Gym, arg0: Asset) -> List[RigidShapeProperties]
    Gets properties for rigid shapes in an asset.
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    obj: list of isaacgym.gymapi.RigidShapeProperties : list of rigid shape properties
    '''


    def get_asset_soft_body_count(self, arg0: Asset) -> int: ...
    ''' get_asset_soft_body_count(self: Gym, arg0: Asset) -> int
    Gets the count of soft bodies on the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    number of soft bodies in asset
    
    Return type:
    int
    '''


    def get_asset_soft_materials(self, arg0: Asset) -> List[SoftMaterial]: ...
    ''' get_asset_soft_materials(self: Gym, arg0: Asset) -> List[SoftMaterial]
    Gets an array of soft materials for the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    list of isaacgym.gymapi.SoftMaterial objects.
    '''


    def get_asset_tendon_count(self, arg0: Asset) -> int: ...
    ''' get_asset_tendon_count(self: Gym, arg0: Asset) -> int
    Gets the count of tendons on a given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    number of tendons in asset
    
    Return type:
    int
    '''


    def get_asset_tendon_name(self, arg0: Asset, arg1: int) -> str: ...
    ''' get_asset_tendon_name(self: Gym, arg0: Asset, arg1: int) -> str
    Gets the name of a tendon of the given asset at the index provided
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - Index of the tendon
    
    
    Returns:
    name of tendon in asset
    
    Return type:
    str
    '''


    def get_asset_tendon_properties(self, arg0: Asset) -> List[TendonProperties]: ...
    ''' get_asset_tendon_properties(self: Gym, arg0: Asset) -> List[TendonProperties]
    Gets an array of tendon properties for the given asset
    Parameters:
    param1 (Asset) - Asset Handle
    
    Returns:
    array containing isaacgym.gymapi.TendonProperties structure
    
    Return type:
    array of isaacgym.gymapi.TendonProperties
    '''


    def get_attractor_properties(self, arg0: Env, arg1: int) -> AttractorProperties: ...
    ''' get_attractor_properties(self: Gym, arg0: Env, arg1: int) -> AttractorProperties
    Get properties of the selected attractor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Attractor) - Attractor Handle
    
    
    Returns:
    properties of selected attractor.
    
    Return type:
    isaacgym.gymapi.AttractorProperties
    '''


    def get_camera_image(self, arg0: Sim, arg1: Env, arg2: int, arg3: ImageType) -> object: ...
    ''' get_camera_image(self: Gym, arg0: Sim, arg1: Env, arg2: int, arg3: ImageType) -> object
    Gets image from selected camera
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Camera) - Camera handle.
    param3 (isaacgym.gymapi.ImageType) - type of image to obtain from camera. see isaacgym.gymapi.ImageType
    
    
    Returns:
    Array containing image data from selected camera
    
    Return type:
    image
    '''


    def get_camera_image_gpu_tensor(self, arg0: Sim, arg1: Env, arg2: int, arg3: ImageType) -> object: ...
    ''' get_camera_image_gpu_tensor(self: Gym, arg0: Sim, arg1: Env, arg2: int, arg3: ImageType) -> object
    Retrieves camera image buffer on GPU
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Env) - Environment Handle.
    param3 (Camera) - Camera handle.
    param4 (isaacgym.gymapi.ImageType) - type of image to obtain from camera. see isaacgym.gymapi.ImageType
    
    
    Returns:
    GymTensor object containing image buffer
    
    Return type:
    isaacgym.gymapi.Tensor
    '''


    def get_camera_proj_matrix(self, arg0: Sim, arg1: Env, arg2: int) -> numpy.ndarray[numpy.float32]: ...
    ''' get_camera_proj_matrix(self: Gym, arg0: Sim, arg1: Env, arg2: int) -> numpy.ndarray[float32]
    Gets Camera Projection Matrix
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Camera) - Camera Handle.
    
    
    Returns:
    Camera projection matrix
    
    Return type:
    isaacgym.gymapi.Mat44
    '''


    def get_camera_transform(self, arg0: Sim, arg1: Env, arg2: int) -> Transform: ...
    ''' get_camera_transform(self: Gym, arg0: Sim, arg1: Env, arg2: int) -> Transform
    Gets Camera Transform
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Env) - refEnv environment of reference to determine the origin
    param3 (Camera) - Camera Handle.
    
    
    Returns:
    Camera Transform
    
    Return type:
    isaacgym.gymapi.Transform
    '''


    def get_camera_view_matrix(self, arg0: Sim, arg1: Env, arg2: int) -> numpy.ndarray[numpy.float32]: ...
    ''' get_camera_view_matrix(self: Gym, arg0: Sim, arg1: Env, arg2: int) -> numpy.ndarray[float32]
    Gets Camera View Matrix
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Camera) - Camera Handle.
    
    
    Returns:
    Camera view matrix
    
    Return type:
    isaacgym.gymapi.Mat44
    '''


    def get_dof_frame(self, arg0: Env, arg1: int) -> DofFrame: ...
    ''' get_dof_frame(self: Gym, arg0: Env, arg1: int) -> DofFrame
    Gets Degree of Freedom Frame
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - Handle to degree of freedom.
    
    
    Returns:
    Frame for DOF
    
    Return type:
    isaacgym.gymapi.DofFrame
    '''


    def get_dof_position(self, arg0: Env, arg1: int) -> float: ...
    ''' get_dof_position(self: Gym, arg0: Env, arg1: int) -> float
    Gets position for a degree of freedom.
    See isaacgym.gymapi.Gym.acquire_dof_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - Handle to degree of freedom.
    
    
    Returns:
    position for DOF
    
    Return type:
    float
    '''


    def get_dof_target_position(self, arg0: Env, arg1: int) -> float: ...
    ''' get_dof_target_position(self: Gym, arg0: Env, arg1: int) -> float
    Gets target position for the DOF. if the DOF is prismatic, the target is in meters. if the DOF is revolute, the target is in radians.
    See isaacgym.gymapi.Gym.set_dof_position_target_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - DOF Handle
    
    
    Returns:
    target position
    
    Return type:
    float
    '''


    def get_dof_target_velocity(self, arg0: Env, arg1: int) -> float: ...
    ''' get_dof_target_velocity(self: Gym, arg0: Env, arg1: int) -> float
    Gets target velocity for the DOF. if the DOF is prismatic, the target is in m/s. if the DOF is revolute, the target is in rad/s.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - DOF Handle
    
    
    Returns:
    target velocity
    
    Return type:
    float
    '''


    def get_dof_type_string(self, arg0: DofType) -> str: ...
    ''' get_dof_type_string(self: Gym, arg0: DofType) -> str
    Converts the degree of freedom type from GymDofType to string
    Parameters:
    param1 (isaacgym.gymapi.DofType) - Degree of freedom type
    
    Returns:
    name of type as string
    
    Return type:
    str
    '''


    def get_dof_velocity(self, arg0: Env, arg1: int) -> float: ...
    ''' get_dof_velocity(self: Gym, arg0: Env, arg1: int) -> float
    Gets velocity for a degree of freedom.
    See isaacgym.gymapi.Gym.acquire_dof_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - Handle to degree of freedom.
    
    
    Returns:
    Velocity for DOF
    
    Return type:
    float
    '''


    def get_elapsed_time(self, arg0: Sim) -> float: ...
    ''' get_elapsed_time(self: Gym, arg0: Sim) -> float
    Gets Elapsed wall clock time since the simulation started.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    elapsed wall clock time, in seconds.
    
    Return type:
    double
    '''


    def get_env(self, arg0: Sim, arg1: int) -> Env: ...
    ''' get_env(self: Gym, arg0: Sim, arg1: int) -> Env
    Gets Environment Handle
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (int) - index of environment.
    
    
    Returns:
    Environment Handle, if index is valid, nullptr otherwise
    
    Return type:
    Env
    '''


    def get_env_count(self, arg0: Sim) -> int: ...
    ''' get_env_count(self: Gym, arg0: Sim) -> int
    Gets count of environments on simulation
    Parameters:
    param1 (Sim) - Simulation Handle.
    
    Returns:
    number of environments
    
    Return type:
    int
    '''


    def get_env_dof_count(self, arg0: Env) -> int: ...
    ''' get_env_dof_count(self: Gym, arg0: Env) -> int
    Gets count of degrees of freedom for environment
    Parameters:
    param1 (Env) - Environment Handle
    
    Returns:
    count of Degrees of Freedom
    
    Return type:
    int
    '''


    def get_env_joint_count(self, arg0: Env) -> int: ...
    ''' get_env_joint_count(self: Gym, arg0: Env) -> int
    Gets count of joints for environment
    Parameters:
    param1 (Env) - Environment Handle
    
    Returns:
    Number of joints in environment
    
    Return type:
    int
    '''


    def get_env_origin(self, arg0: Env) -> Vec3: ...
    ''' get_env_origin(self: Gym, arg0: Env) -> Vec3
    Gets position of environment origin, in simulation global coordinates
    Parameters:
    param1 (Env) - Environment Handle.
    
    Returns:
    environment origin
    
    Return type:
    isaacgym.gymapi.Vec3
    '''


    def get_env_rigid_body_count(self, arg0: Env) -> int: ...
    ''' get_env_rigid_body_count(self: Gym, arg0: Env) -> int
    Gets count of rigid bodies for environment
    Parameters:
    param1 (Env) - Environment Handle
    
    Returns:
    Number of rigid bodies in environment
    
    Return type:
    int
    '''


    def get_env_rigid_body_states(self, arg0: Env, arg1: int) -> numpy.ndarray[RigidBodyState]: ...
    ''' get_env_rigid_body_states(self: Gym, arg0: Env, arg1: int) -> numpy.ndarray[RigidBodyState]
    Gets state for the environments's Rigid Bodies. see isaacgym.gymapi.RigidBodyState.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (int) - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    List of rigid body states
    
    Return type:
    list of isaacgym.gymapi.RigidBodyState
    '''


    def get_env_rigid_contact_forces(self, arg0: Env) -> numpy.ndarray[Vec3]: ...
    ''' get_env_rigid_contact_forces(self: Gym, arg0: Env) -> numpy.ndarray[Vec3]
    Gets contact forces for a single environment.
    See isaacgym.gymapi.Gym.acquire_net_contact_force_tensor for new tensor version of this API.
    Parameters:
    param1 (Env) - environment handle
    
    Returns:
    obj: list of isaacgym.gymapi.Vec3: Contact forces for the environment
    '''


    def get_env_rigid_contacts(self, arg0: Env) -> numpy.ndarray[RigidContact]: ...
    ''' get_env_rigid_contacts(self: Gym, arg0: Env) -> numpy.ndarray[RigidContact]
    Gets contact information for environment
    Parameters:
    param1 (Env) - environment handle
    
    Returns:
    obj: list of isaacgym.gymapi.RigidContact: Contact information for the environment
    '''


    def get_frame_count(self, arg0: Sim) -> int: ...
    ''' get_frame_count(self: Gym, arg0: Sim) -> int
    Gets Current frame count in the simulation.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    current frame count in the simulation
    
    Return type:
    int64_t
    '''


    def get_joint_handle(self, arg0: Env, arg1: str, arg2: str) -> int: ...
    ''' get_joint_handle(self: Gym, arg0: Env, arg1: str, arg2: str) -> int
    Searches for the joint handle in an actor, given their names
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (str) - Actor Name
    param3 (str) - Joint Name
    
    
    Returns:
    Joint handle
    
    Return type:
    Handle
    '''


    def get_joint_name(self, arg0: Env, arg1: int) -> str: ...
    ''' get_joint_name(self: Gym, arg0: Env, arg1: int) -> str
    Gets the joint name.
    Parameters:
    
    param1 (Joint) - Environment handle
    param1 - Joint handle
    
    
    Returns:
    Joint Name
    
    Return type:
    str
    '''


    def get_joint_position(self, arg0: Env, arg1: int) -> float: ...
    ''' get_joint_position(self: Gym, arg0: Env, arg1: int) -> float
    Gets position for the joint.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Handle to joint of interest.
    
    
    Returns:
    The joint position value.
    
    Return type:
    float
    '''


    def get_joint_target_position(self, arg0: Env, arg1: int) -> float: ...
    ''' get_joint_target_position(self: Gym, arg0: Env, arg1: int) -> float
    Gets target position for the joint. if the joint is prismatic, the target is in meters. if the joint is revolute, the target is in radians.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Joint Handle
    
    
    Returns:
    target position
    
    Return type:
    float
    '''


    def get_joint_target_velocity(self, arg0: Env, arg1: int) -> float: ...
    ''' get_joint_target_velocity(self: Gym, arg0: Env, arg1: int) -> float
    Gets target velocity for the joint. if the joint is prismatic, the target is in m/s. if the joint is revolute, the target is in rad/s.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Joint Handle
    
    
    Returns:
    target velocity
    
    Return type:
    float
    '''


    def get_joint_transform(self, arg0: Env, arg1: int) -> Transform: ...
    ''' get_joint_transform(self: Gym, arg0: Env, arg1: int) -> Transform
    Gets Transform for the joint.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Handle to joint of interest.
    
    
    Returns:
    The joint transform value.
    
    Return type:
    isaacgym.gymapi.Transform
    '''


    def get_joint_type_string(self, arg0: JointType) -> str: ...
    ''' get_joint_type_string(self: Gym, arg0: JointType) -> str
    Converts the joint type from JointType to string
    Parameters:
    param1 (JointType) - Type of Joint.
    
    Returns:
    String containing joint type.
    
    Return type:
    string
    '''


    def get_joint_velocity(self, arg0: Env, arg1: int) -> float: ...
    ''' get_joint_velocity(self: Gym, arg0: Env, arg1: int) -> float
    Gets velocity for the joint.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Handle to joint of interest.
    
    
    Returns:
    The joint velocity value.
    
    Return type:
    float
    '''


    def get_performance_timers(self, arg0: Sim, arg1: int) -> PerformanceTimers: ...
    ''' get_performance_timers(self: Gym, arg0: Sim, arg1: int) -> PerformanceTimers
    Returns a struct of performance timers which reflect the times of the most recent operations for a set of timers
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (Timers) - Simulation Timers Handle returned by isaacgym.gymapi.Gym.create_performance_timers
    
    
    Returns:
    Performance timers data structure
    
    Return type:
    isaacgym.gymapi.PerformanceTimers
    '''


    def get_pneumatic_pressure(self, arg0: Env, arg1: int, arg2: int) -> float: ...
    ''' get_pneumatic_pressure(self: Gym, arg0: Env, arg1: int, arg2: int) -> float
    Gets pressure for selected actuator.
    See isaacgym.gymapi.Gym.acquire_pneumatic_pressure_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (SoftActuator) - Soft Actuator Handle
    
    
    Returns:
    pressure, in Pa.
    
    Return type:
    float
    '''


    def get_pneumatic_target(self, arg0: Env, arg1: int, arg2: int) -> float: ...
    ''' get_pneumatic_target(self: Gym, arg0: Env, arg1: int, arg2: int) -> float
    Gets pressure target for selected actuator.
    See isaacgym.gymapi.Gym.acquire_pneumatic_target_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (SoftActuator) - Soft Actuator Handle
    
    
    Returns:
    pressure target, in Pa.
    
    Return type:
    float
    '''


    def get_rigid_angular_velocity(self, arg0: Env, arg1: int) -> Vec3: ...
    ''' get_rigid_angular_velocity(self: Gym, arg0: Env, arg1: int) -> Vec3
    Gets Angular Velocity for Rigid Body.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Body) - Rigid Body handle
    
    
    Returns:
    angular velocity
    
    Return type:
    isaacgym.gymapi.Vec3
    '''


    def get_rigid_body_color(self, arg0: Env, arg1: int, arg2: int, arg3: MeshType) -> Vec3: ...
    ''' get_rigid_body_color(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: MeshType) -> Vec3
    Gets color of rigid body
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (int) - index of rigid body to be set
    param4 (isaacgym.gymapi.MeshType) - selection of what mesh is to be set
    
    
    Returns:
    Vector containing RGB values in range [0,1]
    
    Return type:
    isaacgym.gymapi.Vec3
    '''


    def get_rigid_body_segmentation_id(self, arg0: Env, arg1: int, arg2: int) -> int: ...
    ''' get_rigid_body_segmentation_id(self: Gym, arg0: Env, arg1: int, arg2: int) -> int
    Gets segmentation ID for rigid body
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (int) - index of rigid body to be set
    
    
    Returns:
    segmentation ID for selected body
    
    Return type:
    uint32_t
    '''


    def get_rigid_body_texture(self, arg0: Env, arg1: int, arg2: int, arg3: MeshType) -> int: ...
    ''' get_rigid_body_texture(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: MeshType) -> int
    Gets Handle for rigid body texture
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (int) - index of rigid body to be set
    param4 (isaacgym.gymapi.MeshType) - selection of what mesh is to be set
    
    
    Returns:
    handle for applied texture, INVALID_HANDLE if none is applied.
    
    Return type:
    Handle`
    '''


    def get_rigid_contact_forces(self, arg0: Sim) -> numpy.ndarray[Vec3]: ...
    ''' get_rigid_contact_forces(self: Gym, arg0: Sim) -> numpy.ndarray[Vec3]
    Gets contact forces in simulation.
    See isaacgym.gymapi.Gym.acquire_net_contact_force_tensor for new tensor version of this API.
    Parameters:
    param1 (Sim) - Simulation handle
    
    Returns:
    obj: list of isaacgym.gymapi.Vec3: Contact forces for the simulator
    '''


    def get_rigid_contacts(self, arg0: Sim) -> numpy.ndarray[RigidContact]: ...
    ''' get_rigid_contacts(self: Gym, arg0: Sim) -> numpy.ndarray[RigidContact]
    Gets contact information for simulation.
    Parameters:
    param1 (Sim) - Simulation handle
    
    Returns:
    obj: list of isaacgym.gymapi.RigidContact: Contact information for the simulation
    '''


    def get_rigid_handle(self, arg0: Env, arg1: str, arg2: str) -> int: ...
    ''' get_rigid_handle(self: Gym, arg0: Env, arg1: str, arg2: str) -> int
    Searches for the rigid body in an actor, given their names
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (str) - Actor Name
    param3 (str) - Rigid Body Name
    
    
    Returns:
    Rigid Body handle
    
    Return type:
    Handle
    '''


    def get_rigid_linear_velocity(self, arg0: Env, arg1: int) -> Vec3: ...
    ''' get_rigid_linear_velocity(self: Gym, arg0: Env, arg1: int) -> Vec3
    Gets Linear Velocity for Rigid Body.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Body) - Rigid Body handle
    
    
    Returns:
    linear velocity
    
    Return type:
    isaacgym.gymapi.Vec3
    '''


    def get_rigid_name(self, arg0: Env, arg1: int) -> str: ...
    ''' get_rigid_name(self: Gym, arg0: Env, arg1: int) -> str
    Gets the rigid body name.
    Parameters:
    
    param1 (Body) - Environment handle
    param1 - Rigid Body handle
    
    
    Returns:
    Rigid Body Name
    
    Return type:
    str
    '''


    def get_rigid_transform(self, arg0: Env, arg1: int) -> Transform: ...
    ''' get_rigid_transform(self: Gym, arg0: Env, arg1: int) -> Transform
    Vectorized bindings to get rigid body transforms in the env frame.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Body) - Environment handle
    param1 - Rigid Body handle
    
    
    Returns:
    Transform
    
    Return type:
    isaacgym.gymapi.Transform
    '''


    def get_sensor(self, arg0: Env, arg1: int) -> int: ...
    ''' get_sensor(self: Gym, arg0: Env, arg1: int) -> int
    N/A
    '''


    def get_sim_actor_count(self, arg0: Sim) -> int: ...
    ''' get_sim_actor_count(self: Gym, arg0: Sim) -> int
    Gets total number of actors in simulation
    Parameters:
    param1 (Sim) - Simulation Handle.
    
    Returns:
    Number of actors
    
    Return type:
    int
    '''


    def get_sim_dof_count(self, arg0: Sim) -> int: ...
    ''' get_sim_dof_count(self: Gym, arg0: Sim) -> int
    Gets count of degrees of freedom for simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    count of Degrees of Freedom
    
    Return type:
    int
    '''


    def get_sim_force_sensor_count(self, arg0: Sim) -> int: ...
    ''' get_sim_force_sensor_count(self: Gym, arg0: Sim) -> int
    Gets number of force sensors in simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    Number of sensors
    
    Return type:
    int
    '''


    def get_sim_joint_count(self, arg0: Sim) -> int: ...
    ''' get_sim_joint_count(self: Gym, arg0: Sim) -> int
    Gets count of joints for simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    Joint count for simulation
    
    Return type:
    int
    '''


    def get_sim_params(self, arg0: Sim) -> object: ...
    ''' get_sim_params(self: Gym, arg0: Sim) -> object
    Gets simulation Parameters. See isaacgym.gymapi.SimParams
    Parameters:
    param1 (Sim) - Simulation Handle.
    
    Returns:
    current simulation parameters
    
    Return type:
    isaacgym.gymapi.SimParams
    '''


    def get_sim_rigid_body_count(self, arg0: Sim) -> int: ...
    ''' get_sim_rigid_body_count(self: Gym, arg0: Sim) -> int
    Gets count of Rigid Bodies for simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    Count of rigid bodies
    
    Return type:
    int
    '''


    def get_sim_rigid_body_states(self, arg0: Sim, arg1: int) -> numpy.ndarray[RigidBodyState]: ...
    ''' get_sim_rigid_body_states(self: Gym, arg0: Sim, arg1: int) -> numpy.ndarray[RigidBodyState]
    Gets state for the simulation's Rigid Bodies. see isaacgym.gymapi.RigidBodyState.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (int) - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL),
    positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    List of rigid body states
    
    Return type:
    list of isaacgym.gymapi.RigidBodyState
    '''


    def get_sim_tetrahedra(self, arg0: Sim) -> Tuple[List[int], List[Mat33]]: ...
    ''' get_sim_tetrahedra(self: Gym, arg0: Sim) -> Tuple[List[int], List[Mat33]]
    Gets the tetrahedra indices and Cauchy stress tensors
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    tuple(indices[] (int), stress[] (Matrix33))
    '''


    def get_sim_tetrahedra_count(self, arg0: Sim) -> int: ...
    ''' get_sim_tetrahedra_count(self: Gym, arg0: Sim) -> int
    Gets the total number of tetrahedra in the simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    tetrahedra count
    
    Return type:
    int
    '''


    def get_sim_time(self, arg0: Sim) -> float: ...
    ''' get_sim_time(self: Gym, arg0: Sim) -> float
    Gets Elapsed Simulation Time.
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    elapsed simulation time, in seconds
    
    Return type:
    double
    '''


    def get_sim_triangle_count(self, arg0: Sim) -> int: ...
    ''' get_sim_triangle_count(self: Gym, arg0: Sim) -> int
    Gets the total number of deformable triangles in the simulation
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    triangle count
    
    Return type:
    int
    '''


    def get_sim_triangles(self, arg0: Sim) -> Tuple[List[int], List[int], List[Vec3]]: ...
    ''' get_sim_triangles(self: Gym, arg0: Sim) -> Tuple[List[int], List[int], List[Vec3]]
    Gets the triangle indices, parent tetrahedron indices, and face normals
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    tuple(indices (int[]), parents (int[]), normals (Float3[]))
    '''


    def get_soft_contacts(self, arg0: Sim) -> numpy.ndarray[SoftContact]: ...
    ''' get_soft_contacts(self: Gym, arg0: Sim) -> numpy.ndarray[SoftContact]
    Gets soft contact information for simulation
    Args:
    param1 (Sim): Simulation handle
    Returns:
    :obj: list of isaacgym.gymapi.SoftContact: Contact information for the simulation
    '''


    def get_usd_export_root(self, arg0: UsdExporter) -> str: ...
    ''' get_usd_export_root(self: Gym, arg0: UsdExporter) -> str
    Get USD export directory of a USD exporter
    Parameters:
    param1 (Exporter) - USD Exporter Handle
    
    Returns:
    path the root directory path
    
    Return type:
    str
    '''


    def get_vec_actor_dof_states(self, *args, **kwargs) -> Any: ...
    ''' get_vec_actor_dof_states(*args, **kwargs)
    Overloaded function.
    
    get_vec_actor_dof_states(self: Gym, arg0: List[Env], arg1: int, arg2: int) -> numpy.ndarray[DofState]
    
    
    Gets state for the actor's degrees of freedom from multiple environments. see isaacgym.gymapi.DofState.
    See isaacgym.gymapi.Gym.acquire_dof_state_tensor for new tensor version of this API.
    Args:param1 (:obj: list of Env): Environment Handles
    param2 (Actor): Actor Handle
    param3 (int): flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    Returns:
    Args:param1 (:obj: list of Env): Environment Handles
    param2 (:obj: list of Actor): Actor Handles
    param3 (int): flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    Returns:
    '''


    def get_vec_env_rigid_contact_forces(self, arg0: List[Env]) -> numpy.ndarray[Vec3]: ...
    ''' get_vec_env_rigid_contact_forces(self: Gym, arg0: List[Env]) -> numpy.ndarray[Vec3]
    Gets contact forces for a list of environments.
    See isaacgym.gymapi.Gym.acquire_net_contact_force_tensor for new tensor version of this API.
    Parameters:
    ( (param1) - obj: list of Env): list of environment handles
    
    Returns:
    obj: list of isaacgym.gymapi.Vec3: Contact forces, size of list is numEnvs * numBodies
    '''


    @overload
    def get_vec_rigid_angular_velocity(self, arg0: List[Env], arg1: int) -> numpy.ndarray[Vec3]: ...
    ''' get_vec_rigid_angular_velocity(*args, **kwargs)
    Overloaded function.
    
    get_vec_rigid_angular_velocity(self: Gym, arg0: List[Env], arg1: int) -> numpy.ndarray[Vec3]
    
    vectorized bindings to get rigid body angular velocities in the env frame.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Args:param1 (:obj: list of Env): list of environment handles
    param2 (Body): Rigid Body handle
    
    Returns:

    Args:param1 (:obj: list of Env): list of environment handles
    param2 (:obj: list of Body): list of Rigid Body handles
    
    Returns:
    '''


    @overload
    def get_vec_rigid_angular_velocity(self, arg0: List[Env], arg1: List[int]) -> numpy.ndarray[Vec3]: ...

    @overload
    def get_vec_rigid_linear_velocity(self, arg0: List[Env], arg1: int) -> numpy.ndarray[Vec3]: ...
    ''' get_vec_rigid_linear_velocity(*args, **kwargs)
    Overloaded function.
    
    get_vec_rigid_linear_velocity(self: Gym, arg0: List[Env], arg1: int) -> numpy.ndarray[Vec3]
    
    Vectorized bindings to get rigid body linear velocities in the env frame.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Args:param1 (:obj: list of Env): list of environment handles
    param1 (Body): Rigid Body handle
    
    Returns:

    Args:param1 (:obj: list of Env): list of environment handles
    param1 (:obj: list of Body): list of Rigid Body handles
    
    Returns:
    '''


    @overload
    def get_vec_rigid_linear_velocity(self, arg0: List[Env], arg1: List[int]) -> numpy.ndarray[Vec3]: ...

    @overload
    def get_vec_rigid_transform(self, arg0: List[Env], arg1: int) -> numpy.ndarray[Transform]: ...
    ''' get_vec_rigid_transform(*args, **kwargs)
    Overloaded function.
    
    get_vec_rigid_transform(self: Gym, arg0: List[Env], arg1: int) -> numpy.ndarray[Transform]
    
    Vectorized bindings to get rigid body transforms in the env frame.
    See isaacgym.gymapi.Gym.acquire_rigid_body_state_tensor for new tensor version of this API.
    Args:param1 (:obj: list of Env): list of environment handles
    param1 (Body): Rigid Body handle
    
    Returns:
    
    Args:param1 (:obj: list of Env): list of environment handles
    param1 (:obj: list of Body): list of Rigid Body handles
    
    Returns:
    '''


    @overload
    def get_vec_rigid_transform(self, arg0: List[Env], arg1: List[int]) -> numpy.ndarray[Transform]: ...

    def get_version(self) -> Version: ...
    ''' get_version(self: Gym) -> Version
    Get Gym version.
    Returns:
    isaacgym.gymapi.Version
    '''


    def get_viewer_camera_handle(self, arg0: Viewer) -> int: ...
    ''' get_viewer_camera_handle(self: Gym, arg0: Viewer) -> int
    Gets handle for the viewer camera
    Parameters:
    param1 (Viewer) - Viewer Handle.
    
    Returns:
    Camera Handle
    
    Return type:
    Handle
    '''


    def get_viewer_camera_transform(self, arg0: Viewer, arg1: Env) -> Transform: ...
    ''' get_viewer_camera_transform(self: Gym, arg0: Viewer, arg1: Env) -> Transform
    Gets camera transform, with respect to the selected environment
    Parameters:
    
    param1 (Viewer) - Viewer Handle.
    param2 (Env) - revEnv environment of reference
    
    
    Returns:
    Camera Transform
    
    Return type:
    isaacgym.gymapi.Transform
    '''


    def get_viewer_mouse_position(self, arg0: Viewer) -> Vec2: ...
    ''' get_viewer_mouse_position(self: Gym, arg0: Viewer) -> Vec2
    Returns the latest mouse position from the viewer
    Parameters:
    param1 (Viewer) - Viewer Handle.
    
    Returns:
    Vector containing the normalized mouse coordinates
    
    Return type:
    isaacgym.gymapi.Vec2
    '''


    def get_viewer_size(self, arg0: Viewer) -> Int2: ...
    ''' get_viewer_size(self: Gym, arg0: Viewer) -> Int2
    Returns the size of the window from the viewer
    Parameters:
    param1 (Viewer) - Viewer Handle.
    
    Returns:
    Vector containing the window size
    
    Return type:
    isaacgym.gymapi.Int2
    '''


    def load_asset(self, sim: Sim, rootpath: str, filename: str, options: AssetOptions = ...) -> Asset: ...
    ''' load_asset(self: Gym, sim: Sim, rootpath: str, filename: str, options: AssetOptions = None) -> Asset
    Loads an asset from a file. The file type will be determined by its extension.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (str) - path to asset root folder
    param3 (str) - asset file name
    param4 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def load_mjcf(self, sim: Sim, rootpath: str, filename: str, options: AssetOptions = ...) -> Asset: ...
    ''' load_mjcf(self: Gym, sim: Sim, rootpath: str, filename: str, options: AssetOptions = None) -> Asset
    Loads an Asset from a MJCF file.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (str) - path to asset root folder
    param3 (str) - asset file name
    param4 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def load_opensim(self, sim: Sim, rootpath: str, filename: str, options: AssetOptions = ...) -> Asset: ...
    ''' load_opensim(self: Gym, sim: Sim, rootpath: str, filename: str, options: AssetOptions = None) -> Asset
    Loads an Asset from a OpenSim file.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (str) - path to asset root folder
    param3 (str) - asset file name
    param4 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def load_sim(self, arg0: int, arg1: int, arg2: SimType, arg3: str, arg4: str) -> Sim: ...
    ''' load_sim(self: Gym, arg0: int, arg1: int, arg2: SimType, arg3: str, arg4: str) -> Sim
    Load simulation from USD scene, updates will not be saved to USD stage.
    Parameters:
    
    param1 (int) - index of CUDA-enabled GPU to be used for simulation.
    param2 (int) - index of GPU to be used for rendering.
    param3 (isaacgym.gymapi.SimType) - Type of simulation to be used.
    param4 (string) - Path to root directory of USD scene
    param5 (string) - Filename of USD scene
    
    
    Returns:
    Simulation Handle
    
    Return type:
    Sim
    '''


    def load_urdf(self, sim: Sim, rootpath: str, filename: str, options: AssetOptions = ...) -> Asset: ...
    ''' load_urdf(self: Gym, sim: Sim, rootpath: str, filename: str, options: AssetOptions = None) -> Asset
    Loads an Asset from a URDF file.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (str) - path to asset root folder
    param3 (str) - asset file name
    param4 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def load_usd(self, sim: Sim, rootpath: str, filename: str, options: AssetOptions = ...) -> Asset: ...
    ''' load_usd(self: Gym, sim: Sim, rootpath: str, filename: str, options: AssetOptions = None) -> Asset
    Loads an Asset from a USD file.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (str) - path to asset root folder
    param3 (str) - asset file name
    param4 (isaacgym.gymapi.AssetOptions) - asset Options.
    
    
    Returns:
    Handle to asset
    
    Return type:
    Handle
    '''


    def omni_connect(self, arg0: OmniConnectionParams) -> bool: ...
    ''' omni_connect(self: Gym, arg0: OmniConnectionParams) -> bool
    Connect to Omniverse Kit
    '''


    def omni_disconnect(self) -> None: ...
    ''' omni_disconnect(self: Gym) -> None
    Disconnect from Omniverse Kit
    '''


    def poll_viewer_events(self, arg0: Viewer) -> None: ...
    ''' poll_viewer_events(self: Gym, arg0: Viewer) -> None
    Poll viewer without rendering updates
    Parameters:
    param1 (Viewer) - Viewer Handle.
    '''


    def prepare_sim(self, arg0: Sim) -> bool: ...
    ''' prepare_sim(self: Gym, arg0: Sim) -> bool
    Prepares simulation with buffer allocations
    Parameters:
    param1 (Sim) - Simulation Handle
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def query_viewer_action_events(self, arg0: Viewer) -> List[ActionEvent]: ...
    ''' query_viewer_action_events(self: Gym, arg0: Viewer) -> List[ActionEvent]
    Checks if a given event has occurred.
    Parameters:
    param1 (Viewer) - Viewer Handle.
    
    Returns:
    obj: list of isaacgym.gymapi.ActionEvent: List of events and values
    '''


    def query_viewer_has_closed(self, arg0: Viewer) -> bool: ...
    ''' query_viewer_has_closed(self: Gym, arg0: Viewer) -> bool
    Checks whether the viewer has closed
    Parameters:
    param1 (Viewer) - Viewer Handle.
    
    Returns:
    true if viewer has closed, false otherwise
    
    Return type:
    bool
    '''


    def refresh_actor_root_state_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_actor_root_state_tensor(self: Gym, arg0: Sim) -> bool
    Updates actor root state buffer
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_dof_force_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_dof_force_tensor(self: Gym, arg0: Sim) -> bool
    Updates buffer state for DOF forces
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_dof_state_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_dof_state_tensor(self: Gym, arg0: Sim) -> bool
    Updates DOF state buffer
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_force_sensor_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_force_sensor_tensor(self: Gym, arg0: Sim) -> bool
    Updates buffer state for force sensors tensor
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_jacobian_tensors(self, arg0: Sim) -> bool: ...
    ''' refresh_jacobian_tensors(self: Gym, arg0: Sim) -> bool
    Updates buffer state for Jacobian tensor
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_mass_matrix_tensors(self, arg0: Sim) -> bool: ...
    ''' refresh_mass_matrix_tensors(self: Gym, arg0: Sim) -> bool
    Updates buffer state for Mass Matrix tensor
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_net_contact_force_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_net_contact_force_tensor(self: Gym, arg0: Sim) -> bool
    Updates buffer state for net contact force tensor
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_particle_state_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_particle_state_tensor(self: Gym, arg0: Sim) -> bool
    Updates buffer for particle states tensor. Flex backend only.
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_pneumatic_pressure_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_pneumatic_pressure_tensor(self: Gym, arg0: Sim) -> bool
    Updates buffer state for pneumatic pressure tensor. Flex backend only.
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_pneumatic_target_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_pneumatic_target_tensor(self: Gym, arg0: Sim) -> bool
    Updates buffer state for pneumatic target tensor. Flex backend only.
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def refresh_rigid_body_state_tensor(self, arg0: Sim) -> bool: ...
    ''' refresh_rigid_body_state_tensor(self: Gym, arg0: Sim) -> bool
    Updates rigid body states buffer
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def render_all_camera_sensors(self, arg0: Sim) -> None: ...
    ''' render_all_camera_sensors(self: Gym, arg0: Sim) -> None
    Renders all images obtained from camera sensors
    Parameters:
    param1 (Sim) - Simulation Handle.
    '''


    def reset_actor_materials(self, arg0: Env, arg1: int, arg2: MeshType) -> None: ...
    ''' reset_actor_materials(self: Gym, arg0: Env, arg1: int, arg2: MeshType) -> None
    Resets all materials on an actor to what was loaded with the asset file.
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (isaacgym.gymapi.MeshType) - selection of what mesh is to be set
    '''


    def reset_actor_particles_to_rest(self, arg0: Env, arg1: int) -> bool: ...
    ''' reset_actor_particles_to_rest(self: Gym, arg0: Env, arg1: int) -> bool
    Resets particles in actor to their rest position
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_dof_position_targets(self, *args, **kwargs) -> Any: ...
    ''' set_actor_dof_position_targets(self: Gym, arg0: Env, arg1: int, arg2: numpy.ndarray[float32]) -> bool
    
    Sets target position for the actor's degrees of freedom. if the joint is prismatic, the target is in meters. if the joint is revolute, the target is in radians.
    See isaacgym.gymapi.Gym.set_dof_position_target_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of float) : array of position targets
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_dof_properties(self, arg0: Env, arg1: int, arg2) -> bool: ...
    ''' set_actor_dof_properties(self: Gym, arg0: Env, arg1: int, arg2: numpy.ndarray[carb::gym::GymDofProperties]) -> bool
    Sets Degrees of Freedom Properties for Actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    properties. (param3 A structured Numpy array of DOF) - 
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_dof_states(self, arg0: Env, arg1: int, arg2: numpy.ndarray[DofState], arg3: int) -> bool: ...
    ''' set_actor_dof_states(self: Gym, arg0: Env, arg1: int, arg2: numpy.ndarray[DofState], arg3: int) -> bool
    Sets states for the actor's Rigid Bodies. see isaacgym.gymapi.RigidBodyState.
    See isaacgym.gymapi.Gym.set_dof_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of isaacgym.gymapi.DofState): array of states to set.
    param4 (int) - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_dof_velocity_targets(self, *args, **kwargs) -> Any: ...
    ''' set_actor_dof_velocity_targets(self: Gym, arg0: Env, arg1: int, arg2: numpy.ndarray[float32]) -> bool
    
    Sets target velocities for the actor's degrees of fredom. if the joint is prismatic, the target is in m/s. if the joint is revolute, the target is in rad/s.
    See isaacgym.gymapi.Gym.set_dof_velocity_target_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of float) : array of velocity targets
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_fixed_tendon_joint_coefficients(self, arg0: Env, arg1: int, arg2: int, arg3: List[float]) -> bool: ...
    ''' set_actor_fixed_tendon_joint_coefficients(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: List[float]) -> bool
    Sets coefficients of joints in fixed tendon of an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of fixed tendon
    ( (param4) - obj: list of float): list of coefficients to set
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_rigid_body_properties(self, env: Env, actorHandle: int, props: List[RigidBodyProperties],
                                        recomputeInertia: bool = ...) -> bool: ...
    ''' set_actor_rigid_body_properties(self: Gym, env: Env, actorHandle: int, props: List[RigidBodyProperties], recomputeInertia: bool = False) -> bool
    Sets properties for rigid bodies in an actor on selected environment.
    Note: Changing the center-of-mass when using the GPU pipeline is currently not supported (but mass and inertia can be changed).
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of isaacgym.gymapi.RigidBodyProperties): list of properties to set
    param4 (bool) - flag for recomputing inertia tensor on a mass change
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_rigid_body_states(self, arg0: Env, arg1: int, arg2: numpy.ndarray[RigidBodyState],
                                    arg3: int) -> bool: ...
    ''' set_actor_rigid_body_states(self: Gym, arg0: Env, arg1: int, arg2: numpy.ndarray[RigidBodyState], arg3: int) -> bool
    Sets states for the actor's Rigid Bodies. see isaacgym.gymapi.RigidBodyState.
    See isaacgym.gymapi.Gym.set_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of isaacgym.gymapi.RigidBodyState): array of states to set.
    param4 (int) - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_rigid_shape_properties(self, arg0: Env, arg1: int, arg2: List[RigidShapeProperties]) -> bool: ...
    ''' set_actor_rigid_shape_properties(self: Gym, arg0: Env, arg1: int, arg2: List[RigidShapeProperties]) -> bool
    Sets properties for rigid shapes in an ector on selected environment.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of isaacgym.gymapi.RigidShapeProperties): list of properties to set
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_root_state_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_actor_root_state_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets actor root state buffer to values provided for all actors in simulation.
    See isaacgym.gymapi.Gym.set_actor_root_state_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing actor root states
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_root_state_tensor_indexed(self, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool: ...
    ''' set_actor_root_state_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool
    Sets actor root state buffer to values provided for given actor indices.
    Full actor root states buffer should be provided for all actors.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing actor root states
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_scale(self, arg0: Env, arg1: int, arg2: float) -> bool: ...
    ''' set_actor_scale(self: Gym, arg0: Env, arg1: int, arg2: float) -> bool
    Sets the scale of the actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (float) - 
    
    
    Returns:
    True if scale was set successfully, false otherwise.
    '''


    def set_actor_soft_materials(self, arg0: Env, arg1: int, arg2: List[SoftMaterial]) -> bool: ...
    ''' set_actor_soft_materials(self: Gym, arg0: Env, arg1: int, arg2: List[SoftMaterial]) -> bool
    Sets soft material properties for Actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of isaacgym.gymapi.SoftMaterials):  properties for soft materials, see isaacgym.gymapi.SoftMaterials.
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_actor_tendon_offset(self, arg0: Env, arg1: int, arg2: int, arg3: float) -> bool: ...
    ''' set_actor_tendon_offset(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: float) -> bool
    Sets the length offset of a tendon of an actor
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (int) - Index of the tendon
    param4 (float) - The offset to set
    
    
    Returns:
    indicator whether operation was successful
    
    Return type:
    bool
    '''


    def set_actor_tendon_properties(self, arg0: Env, arg1: int, arg2: List[TendonProperties]) -> bool: ...
    ''' set_actor_tendon_properties(self: Gym, arg0: Env, arg1: int, arg2: List[TendonProperties]) -> bool
    Sets properties for tendons in an actor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    ( (param3) - obj: list of isaacgym.gymapi.TendonProperties): list of properties to set
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_asset_fixed_tendon_joint_coefficients(self, arg0: Asset, arg1: int, arg2: List[float]) -> bool: ...
    ''' set_asset_fixed_tendon_joint_coefficients(self: Gym, arg0: Asset, arg1: int, arg2: List[float]) -> bool
    Sets coefficients of joints in fixed tendon in an asset.
    Parameters:
    
    param1 (Asset) - Asset Handle
    param2 (int) - Index of fixed tendon
    ( (param3) - obj: list of float): list of coefficients to set
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_asset_rigid_shape_properties(self, arg0: Asset, arg1: List[RigidShapeProperties]) -> bool: ...
    ''' set_asset_rigid_shape_properties(self: Gym, arg0: Asset, arg1: List[RigidShapeProperties]) -> bool
    Sets properties for rigid shapes in an asset.
    Parameters:
    
    param1 (Asset) - Asset Handle
    ( (param2) - obj: list of isaacgym.gymapi.RigidShapeProperties): list of properties to set
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_asset_tendon_properties(self, arg0: Asset, arg1: List[TendonProperties]) -> bool: ...
    ''' set_asset_tendon_properties(self: Gym, arg0: Asset, arg1: List[TendonProperties]) -> bool
    Sets properties for tendons in an asset.
    Parameters:
    
    param1 (Asset) - Asset Handle
    ( (param2) - obj: list of isaacgym.gymapi.TendonProperties): list of properties to set
    
    
    Returns:
    return true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_attractor_properties(self, arg0: Env, arg1: int, arg2: AttractorProperties) -> None: ...
    ''' set_attractor_properties(self: Gym, arg0: Env, arg1: int, arg2: AttractorProperties) -> None
    Modifies the properties of an attractor given by its handle and the environment selected. for modifying only the attractor target, see setAttractorTarget.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Attractor) - Attractor Handle
    param3 (isaacgym.gymapi.AttractorProperties) - the new properties for the attractor.
    '''


    def set_attractor_target(self, arg0: Env, arg1: int, arg2: Transform) -> None: ...
    ''' set_attractor_target(self: Gym, arg0: Env, arg1: int, arg2: Transform) -> None
    Modifies target of the selected attractor.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Attractor) - Attractor Handle
    param3 (isaacgym.gymapi.Transform) - attractor target pose, in global coordinate.
    
    
    Returns:
    properties of selected attractor.
    
    Return type:
    isaacgym.gymapi.AttractorProperties
    '''


    def set_camera_location(self, arg0: int, arg1: Env, arg2: Vec3, arg3: Vec3) -> None: ...
    ''' set_camera_location(self: Gym, arg0: int, arg1: Env, arg2: Vec3, arg3: Vec3) -> None
    Positions the viewer camera to look at a specified target location. Setting Camera location manually will detach the camera if it is attached to a body
    Parameters:
    
    param1 (Camera) - Viewer Handle.
    param2 (Env) - refEnv environment of reference to determine the origin
    param3 (isaacgym.gymapi.Vec3) - position where the camera will be placed, with respect to the selected environment origin
    param4 (isaacgym.gymapi.Vec3) - target location that will be at the center of the camera, with respect to the selected environment
    
    
    Returns:
    true if viewer has closed, false otherwise
    
    Return type:
    bool
    '''


    def set_camera_transform(self, arg0: int, arg1: Env, arg2: Transform) -> None: ...
    ''' set_camera_transform(self: Gym, arg0: int, arg1: Env, arg2: Transform) -> None
    Sets camera transform with respect to the attached rigid body.
    Parameters:
    
    param1 (Camera) - Viewer Handle.
    param2 (Env) - refEnv environment of reference to determine the origin
    param3 (isaacgym.gymapi.Transform) - transform from rigid body to camera position
    
    
    Returns:
    true if viewer has closed, false otherwise
    
    Return type:
    bool
    '''


    def set_dof_actuation_force_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_dof_actuation_force_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets DOF actuation forces to values provided for all DOFs in simulation.
    Force is in Newton for linear DOF. For revolute DOF, force is torque in Nm.
    See isaacgym.gymapi.Gym.set_dof_actuation_force_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing forces
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_actuation_force_tensor_indexed(self, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool: ...
    ''' set_dof_actuation_force_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool
    Sets DOF actuation forces to values provided for given actor indices.
    Full DOF actuation force buffer should be provided for all actors.
    Force is in Newton for linear DOF. For revolute DOF, force is torque in Nm.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF actuations states
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_position_target_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_dof_position_target_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets DOF position targets to values provided for all DOFs in simulation.
    For presimatic DOF, target is in meters. For revolute DOF, target is in radians.
    See isaacgym.gymapi.Gym.set_dof_position_target_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF position targets
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_position_target_tensor_indexed(self, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool: ...
    ''' set_dof_position_target_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool
    Sets DOF position targets to values provided for given actor indices.
    Full DOF position targets buffer should be provided for all actors.
    For presimatic DOF, target is in meters. For revolute DOF, target is in radians.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF position targets
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_state_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_dof_state_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets DOF state buffer to values provided for all DOFs in simulation.
    DOF state includes position in meters for prismatic DOF, or radians for revolute DOF, and velocity in m/s for prismatic DOF and rad/s for revolute DOF.
    See isaacgym.gymapi.Gym.set_dof_state_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF states
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_state_tensor_indexed(self, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool: ...
    ''' set_dof_state_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool
    Sets DOF state buffer to values provided for given actor indices.
    Full DOF state buffer should be provided for all actors.
    DOF state includes position in meters for prismatic DOF, or radians for revolute DOF, and velocity in m/s for prismatic DOF and rad/s for revolute DOF.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF states
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_target_position(self, arg0: Env, arg1: int, arg2: float) -> None: ...
    ''' set_dof_target_position(self: Gym, arg0: Env, arg1: int, arg2: float) -> None
    Sets a target position for the DOF. if the DOF is prismatic, the target is in meters. if the DOF is revolute, the target is in radians.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - DOF Handle
    param3 (float) - position
    '''


    def set_dof_target_velocity(self, arg0: Env, arg1: int, arg2: float) -> None: ...
    ''' set_dof_target_velocity(self: Gym, arg0: Env, arg1: int, arg2: float) -> None
    Sets a target velocity for the DOF. if the DOF is prismatic, the target is in m/s. if the DOF is revolute, the target is in rad/s.
    See isaacgym.gymapi.Gym.set_dof_velocity_target_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (DOF) - DOF Handle
    param3 (float) - velocity
    '''


    def set_dof_velocity_target_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_dof_velocity_target_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets DOF velocity targets to values provided for all DOFs in simulation
    For prismatic DOF, target is in m/s. For revolute DOF, target is in rad/s.
    See isaacgym.gymapi.Gym.set_dof_velocity_target_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF velocity targets
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_dof_velocity_target_tensor_indexed(self, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool: ...
    ''' set_dof_velocity_target_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool
    Sets DOF velocity targets to values provided for given actor indices.
    Full DOF velocity target buffer should be provided for all actors.
    For prismatic DOF, target is in m/s. For revolute DOF, target is in rad/s.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing DOF velocity targets
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_env_rigid_body_states(self, arg0: Env, arg1: numpy.ndarray[RigidBodyState], arg2: int) -> bool: ...
    ''' set_env_rigid_body_states(self: Gym, arg0: Env, arg1: numpy.ndarray[RigidBodyState], arg2: int) -> bool
    Sets states for simulation's Rigid Bodies. see isaacgym.gymapi.RigidBodyState
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (int) - List of rigid body states
    param2 - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL), positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_joint_target_position(self, arg0: Env, arg1: int, arg2: float) -> None: ...
    ''' set_joint_target_position(self: Gym, arg0: Env, arg1: int, arg2: float) -> None
    Sets a target position for the joint. if the joint is prismatic, the target is in meters. if the joint is revolute, the target is in radians.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Joint Handle
    param3 (float) - position
    '''


    def set_joint_target_velocity(self, arg0: Env, arg1: int, arg2: float) -> None: ...
    ''' set_joint_target_velocity(self: Gym, arg0: Env, arg1: int, arg2: float) -> None
    Sets a target velocity for the joint. if the joint is prismatic, the target is in m/s. if the joint is revolute, the target is in rad/s.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Joint) - Joint Handle
    param3 (float) - velocity
    '''


    def set_light_parameters(self, arg0: Sim, arg1: int, arg2: Vec3, arg3: Vec3, arg4: Vec3) -> None: ...
    ''' set_light_parameters(self: Gym, arg0: Sim, arg1: int, arg2: Vec3, arg3: Vec3, arg4: Vec3) -> None
    Sets light parameters
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (int) - index to the light to be changed.
    param3 (isaacgym.gymapi.Vec3) - intensity intensity of the light focus in the range [0,1] per channel, in RGB.
    param4 (isaacgym.gymapi.Vec3) - ambient intensity of the ambient light in the range [0,1] per channel, in RGB.
    param5 (isaacgym.gymapi.Vec3) - direction direction of the light focus
    '''


    def set_particle_state_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_particle_state_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets particle states buffer to values provided. Flex backend only.
    See isaacgym.gymapi.Gym.set_particle_state_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing particle states
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_particle_state_tensor_indexed(self, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool: ...
    ''' set_particle_state_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor, arg2: Tensor, arg3: int) -> bool
    Sets particle state buffer to values provided for given actor indices. Flex backend only.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing particle states
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_pneumatic_pressure(self, arg0: Env, arg1: int, arg2: int, arg3: float) -> None: ...
    ''' set_pneumatic_pressure(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: float) -> None
    Sets pressure for selected actuator. the pressure value is clamped between 0 and actuator max pressure.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (SoftActuator) - Soft Actuator Handle
    param4 (float) - pressure, in Pa.
    '''


    def set_pneumatic_pressure_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_pneumatic_pressure_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets pneumatic pressure buffer to values provided. Flex backend only.
    See isaacgym.gymapi.Gym.set_pneumatic_pressure_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing pneumatic pressures
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_pneumatic_pressure_tensor_indexed(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_pneumatic_pressure_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets pneumatic pressure buffer to values provided for given actor indices. Flex backend only.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing pneumatic pressure
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_pneumatic_target(self, arg0: Env, arg1: int, arg2: int, arg3: float) -> None: ...
    ''' set_pneumatic_target(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: float) -> None
    Sets target pressure for selected actuator. the target value is clamped between 0 and actuator max pressure.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Actor) - Actor Handle
    param3 (SoftActuator) - Soft Actuator Handle
    param4 (float) - pressure target, in Pa.
    '''


    def set_pneumatic_target_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_pneumatic_target_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets pneumatic target buffer to values provided. Flex backend only.
    See isaacgym.gymapi.Gym.set_pneumatic_target_tensor_indexed for indexed version of this API.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing pneumatic targets
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_pneumatic_target_tensor_indexed(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_pneumatic_target_tensor_indexed(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets pneumatic target buffer to values provided for given actor indices. Flex backend only.
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing pneumatic targets
    param3 (isaacgym.gymapi.Tensor) - Buffer containing actor indices
    param4 (int) - Size of actor indices buffer
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_rigid_angular_velocity(self, arg0: Env, arg1: int, arg2: Vec3) -> None: ...
    ''' set_rigid_angular_velocity(self: Gym, arg0: Env, arg1: int, arg2: Vec3) -> None
    Sets Angular Velocity for Rigid Body.
    See isaacgym.gymapi.Gym.set_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Body) - Rigid Body Handle
    param3 (isaacgym.gymapi.Vec3) - angular velocity.
    '''


    def set_rigid_body_color(self, arg0: Env, arg1: int, arg2: int, arg3: MeshType, arg4: Vec3) -> None: ...
    ''' set_rigid_body_color(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: MeshType, arg4: Vec3) -> None
    Sets color of rigid body
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (int) - index of rigid body to be set
    param4 (isaacgym.gymapi.MeshType) - selection of what mesh is to be set
    param5 (isaacgym.gymapi.Vec3) - Vector containing RGB values in range [0,1]
    '''


    def set_rigid_body_segmentation_id(self, arg0: Env, arg1: int, arg2: int, arg3: int) -> None: ...
    ''' set_rigid_body_segmentation_id(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: int) -> None
    Sets segmentation ID for rigid body
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (int) - index of rigid body to be set
    param4 (uint32_t) - segmentation ID for selected body
    '''


    def set_rigid_body_state_tensor(self, arg0: Sim, arg1: Tensor) -> bool: ...
    ''' set_rigid_body_state_tensor(self: Gym, arg0: Sim, arg1: Tensor) -> bool
    Sets rigid body state buffer to values provided for all rigid bodies in simulation. Flex backend only.
    State for each rigid body should contain position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (isaacgym.gymapi.Tensor) - Buffer containing rigid body states
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_rigid_body_texture(self, arg0: Env, arg1: int, arg2: int, arg3: MeshType, arg4: int) -> None: ...
    ''' set_rigid_body_texture(self: Gym, arg0: Env, arg1: int, arg2: int, arg3: MeshType, arg4: int) -> None
    Sets texture on Rigid Body
    Parameters:
    
    param1 (Env) - Environment Handle.
    param2 (Actor) - Actor Handle
    param3 (int) - index of rigid body to be set
    param4 (isaacgym.gymapi.MeshType) - selection of what mesh is to be set
    param5 (Texture) - texture handle for the selected texture
    '''


    def set_rigid_linear_velocity(self, *args, **kwargs) -> Any: ...
    ''' set_rigid_linear_velocity(self: Gym, arg0: Env, arg1: int, arg2: Vec3) -> None
    
    Sets Linear Velocity for Rigid Body.
    See isaacgym.gymapi.Gym.set_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Body) - Rigid Body handle
    param3 (isaacgym.gymapi.Vec3) - linear velocity
    '''


    def set_rigid_transform(self, *args, **kwargs) -> Any: ...
    ''' set_rigid_transform(self: Gym, arg0: Env, arg1: int, arg2: Transform) -> None
    
    Sets Transform for Rigid Body.
    See isaacgym.gymapi.Gym.set_rigid_body_state_tensor for new tensor version of this API.
    Parameters:
    
    param1 (Env) - Environment Handle
    param2 (Body) - Rigid Body handle
    param3 (isaacgym.gymapi.Transform) - Transform
    '''


    def set_sim_device(self, arg0: Sim) -> None: ...
    ''' set_sim_device(self: Gym, arg0: Sim) -> None
    Sets sim compute device to be the current CUDA device.
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def set_sim_params(self, arg0: Sim, arg1: SimParams) -> None: ...
    ''' set_sim_params(self: Gym, arg0: Sim, arg1: SimParams) -> None
    Sets simulation Parameters. See isaacgym.gymapi.SimParams
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (isaacgym.gymapi.SimParams) - simulation parameters
    '''


    def set_sim_rigid_body_states(self, arg0: Sim, arg1: numpy.ndarray[RigidBodyState], arg2: int) -> bool: ...
    ''' set_sim_rigid_body_states(self: Gym, arg0: Sim, arg1: numpy.ndarray[RigidBodyState], arg2: int) -> bool
    Sets states for simulation's Rigid Bodies. see isaacgym.gymapi.RigidBodyState
    Parameters:
    
    param1 (Sim) - Simulation Handle
    param2 (int) - List of rigid body states
    param2 - flags for the state to obtain, can be velocities (isaacgym.gymapi.STATE_VEL),
    positions(isaacgym.gymapi.STATE_POS) or both (isaacgym.gymapi.STATE_ALL)
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def set_usd_export_root(self, arg0: UsdExporter, arg1: str) -> bool: ...
    ''' set_usd_export_root(self: Gym, arg0: UsdExporter, arg1: str) -> bool
    Set USD export directory of a USD exporter
    Parameters:
    
    param1 (Exporter) - USD Exporter Handle
    param3 (str) - path the root directory path (pass “.” for current working directory)
    
    
    Returns:
    true if operation was succesful, false otherwise.
    
    Return type:
    bool
    '''


    def simulate(self, arg0: Sim) -> None: ...
    ''' simulate(self: Gym, arg0: Sim) -> None
    Steps the simulation by one time-step of dt, in seconds, divided in n substeps
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def start_access_image_tensors(self, arg0: Sim) -> None: ...
    ''' start_access_image_tensors(self: Gym, arg0: Sim) -> None
    Allows access to image tensors. Transfers data from all image tensors from the GPU to memory
    Parameters:
    param1 (Sim) - Simulation Handle.
    '''


    def step_graphics(self, arg0: Sim) -> None: ...
    ''' step_graphics(self: Gym, arg0: Sim) -> None
    Update graphics of the simulator
    Updates the simulation's graphics. If one is displaying the simulation through a viewer, this method should be called in advance to obtain the latest graphics state.
    Parameters:
    param1 (Sim) - Simulation Handle.
    '''


    def subscribe_viewer_keyboard_event(self, arg0: Viewer, arg1: KeyboardInput, arg2: str) -> None: ...
    ''' subscribe_viewer_keyboard_event(self: Gym, arg0: Viewer, arg1: KeyboardInput, arg2: str) -> None
    Subscribes an action to a keyboard event. Each action keeps a list of mappings. This function push the mapping to the end of the list.
    Parameters:
    
    param1 (Viewer) - Viewer Handle.
    param2 (isaacgym.gymapi.KeyboardInput) - keyboard input to subscribe to
    param3 (str) - name of the action to be mapped
    '''


    def subscribe_viewer_mouse_event(self, arg0: Viewer, arg1: MouseInput, arg2: str) -> None: ...
    ''' subscribe_viewer_mouse_event(self: Gym, arg0: Viewer, arg1: MouseInput, arg2: str) -> None
    Subscribes an action to a mouse event. Each action keeps a list of mappings. This function push the mapping to the end of the list.
    Parameters:
    
    param1 (Viewer) - Viewer Handle.
    param2 (isaacgym.gymapi.MouseInput) - mouse input to subscribe to
    param3 (str) - name of the action to be mapped
    '''


    def sync_frame_time(self, arg0: Sim) -> None: ...
    ''' sync_frame_time(self: Gym, arg0: Sim) -> None
    Throttles simulation speed to real time.
    Parameters:
    param1 (Sim) - Simulation Handle
    '''


    def viewer_camera_look_at(self, arg0: Viewer, arg1: Env, arg2: Vec3, arg3: Vec3) -> None: ...
    ''' viewer_camera_look_at(self: Gym, arg0: Viewer, arg1: Env, arg2: Vec3, arg3: Vec3) -> None
    Positions the viewer camera to look at a specified target location
    Parameters:
    
    param1 (Viewer) - Viewer Handle.
    param2 (Env) - refEnv environment of reference to determine the origin
    param3 (isaacgym.gymapi.Vec3) - position where the camera will be placed, with respect to the selected environment origin
    param4 (isaacgym.gymapi.Vec3) - target location that will be at the center of the camera, with respect to the selected environment
    '''


    def write_camera_image_to_file(self, arg0: Sim, arg1: Env, arg2: int, arg3: ImageType, arg4: str) -> None: ...
    ''' write_camera_image_to_file(self: Gym, arg0: Sim, arg1: Env, arg2: int, arg3: ImageType, arg4: str) -> None
    Writes image from camera directly to a PNG file.
    Parameters:
    
    param1 (Sim) - Simulation Handle.
    param2 (Camera) - Camera Handle
    param3 (isaacgym.gymapi.ImageType) - type of image to obtain from camera. see isaacgym.gymapi.ImageType
    param4 (str) - filename string with the name of the file to be saved
    '''


    def write_viewer_image_to_file(self, arg0: Viewer, arg1: str) -> None: ...
    ''' write_viewer_image_to_file(self: Gym, arg0: Viewer, arg1: str) -> None
    Outputs image obtained from a viewer directly to a PNG file.
    Parameters:
    
    param1 (Viewer) - Viewer Handle.
    param2 (str) - filename string with the name of the file to be saved
    '''



class HeightFieldParams:
    ''' class isaacgym.gymapi.HeightFieldParams
    The heightfield origin is at its center (height = 0), and it is oriented to be perpendicular to the the Gym up-axis.
    '''

    column_scale: float  # property column_scale Spacing of samples [m] in column dimension
    dynamic_friction: float  # property dynamic_friction Coefficient of dynamic friction
    nbColumns: int  # property nbColumns -Y) Type: Number of samples in column dimension (Y-up  Type: Z, Z-up
    nbRows: int  # property nbRows Number of samples in row dimension (X)
    restitution: float  # property restitution Coefficient of restitution
    row_scale: float  # property row_scale Spacing of samples [m] in row dimension
    segmentation_id: int  # property segmentation_id SegmentationID value for segmentation ground truth
    static_friction: float  # property static_friction Coefficient of static friction
    transform: Transform  # property transform Transform to apply to heightfield
    vertical_scale: float  # property vertical_scale Vertical scaling [m] to apply to integer height samples

    def __init__(self) -> None: ...


class ImageType:
    ''' class isaacgym.gymapi.ImageType
    Types of image generated by the sensors
    Members:
    
    IMAGE_COLOR : Image RGB. Regular image as a camera sensor would generate. Each pixel is made of three values of the selected data type GymTensorDataType, representing the intensity of Red, Green and Blue.
    IMAGE_DEPTH : Depth Image. Each pixel is one value of the selected data type GymTensorDataType, representing how far that point is from the center of the camera.
    IMAGE_SEGMENTATION : Segmentation Image. Each pixel is one integer value from the selected data type GymTensorDataType, that represents the class of the object that is displayed on that pixel
    IMAGE_OPTICAL_FLOW : Optical Flow image - each pixel is a 2D vector of the screen-space velocity of the bodies visible in that pixel
    '''

    __members__: ClassVar[dict] = ...  # read-only
    IMAGE_COLOR: ClassVar[ImageType] = ...
    IMAGE_DEPTH: ClassVar[ImageType] = ...
    IMAGE_OPTICAL_FLOW: ClassVar[ImageType] = ...
    IMAGE_SEGMENTATION: ClassVar[ImageType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class IndexDomain:
    ''' class isaacgym.gymapi.IndexDomain
    Domain type for indexing into component buffers.
    Members:
    
    DOMAIN_ACTOR
    DOMAIN_ENV
    DOMAIN_SIM
    '''

    __members__: ClassVar[dict] = ...  # read-only
    DOMAIN_ACTOR: ClassVar[IndexDomain] = ...
    DOMAIN_ENV: ClassVar[IndexDomain] = ...
    DOMAIN_SIM: ClassVar[IndexDomain] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class IndexRange:
    ''' class isaacgym.gymapi.IndexRange
    Used for passing start and end indexes of a vector when setting or getting data of a slice of the vector.
    '''

    count: int  # property count
    start: int  # property start

    def __init__(self) -> None: ...


class Int2:
    dtype: ClassVar[dtype] = ...  # read-only
    x: int
    y: int

    def __init__(self, x: int = ..., y: int = ...) -> None: ...


class Int3:
    dtype: ClassVar[dtype] = ...  # read-only
    x: int
    y: int
    z: int

    def __init__(self, x: int = ..., y: int = ..., z: int = ...) -> None: ...


class JointType:
    ''' class isaacgym.gymapi.JointType
    Types of Joint supported by the simulator
    Members:
    
    JOINT_INVALID : invalid/unknown/uninitialized joint type.
    JOINT_FIXED : Fixed joint. Bodies will move together.
    JOINT_REVOLUTE : Revolute or Hinge Joint. Bodies will rotate on one defined axis.
    JOINT_PRISMATIC : Prismatic Joints. Bodies will move linearly on one axis.
    JOINT_BALL : Ball Joint. Bodies will rotate on all directions on point of reference.
    JOINT_PLANAR : Planar Joint. Bodies will move on defined plane.
    JOINT_FLOATING : Floating Joint. No constraints added between bodies.
    '''

    __members__: ClassVar[dict] = ...  # read-only
    JOINT_BALL: ClassVar[JointType] = ...
    JOINT_FIXED: ClassVar[JointType] = ...
    JOINT_FLOATING: ClassVar[JointType] = ...
    JOINT_INVALID: ClassVar[JointType] = ...
    JOINT_PLANAR: ClassVar[JointType] = ...
    JOINT_PRISMATIC: ClassVar[JointType] = ...
    JOINT_REVOLUTE: ClassVar[JointType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class KeyboardInput:
    ''' class isaacgym.gymapi.KeyboardInput
    Members:
    KEY_SPACE
    KEY_APOSTROPHE
    KEY_COMMA
    KEY_MINUS
    KEY_PERIOD
    KEY_SLASH
    KEY_0
    KEY_1
    KEY_2
    KEY_3
    KEY_4
    KEY_5
    KEY_6
    KEY_7
    KEY_8
    KEY_9
    KEY_SEMICOLON
    KEY_EQUAL
    KEY_A
    KEY_B
    KEY_C
    KEY_D
    KEY_E
    KEY_F
    KEY_G
    KEY_H
    KEY_I
    KEY_J
    KEY_K
    KEY_L
    KEY_M
    KEY_N
    KEY_O
    KEY_P
    KEY_Q
    KEY_R
    KEY_S
    KEY_T
    KEY_U
    KEY_V
    KEY_W
    KEY_X
    KEY_Y
    KEY_Z
    KEY_LEFT_BRACKET
    KEY_BACKSLASH
    KEY_RIGHT_BRACKET
    KEY_GRAVE_ACCENT
    KEY_ESCAPE
    KEY_TAB
    KEY_ENTER
    KEY_BACKSPACE
    KEY_INSERT
    KEY_DEL
    KEY_RIGHT
    KEY_LEFT
    KEY_DOWN
    KEY_UP
    KEY_PAGE_UP
    KEY_PAGE_DOWN
    KEY_HOME
    KEY_END
    KEY_CAPS_LOCK
    KEY_SCROLL_LOCK
    KEY_NUM_LOCK
    KEY_PRINT_SCREEN
    KEY_PAUSE
    KEY_F1
    KEY_F2
    KEY_F3
    KEY_F4
    KEY_F5
    KEY_F6
    KEY_F7
    KEY_F8
    KEY_F9
    KEY_F10
    KEY_F11
    KEY_F12
    KEY_NUMPAD_0
    KEY_NUMPAD_1
    KEY_NUMPAD_2
    KEY_NUMPAD_3
    KEY_NUMPAD_4
    KEY_NUMPAD_5
    KEY_NUMPAD_6
    KEY_NUMPAD_7
    KEY_NUMPAD_8
    KEY_NUMPAD_9
    KEY_NUMPAD_DEL
    KEY_NUMPAD_DIVIDE
    KEY_NUMPAD_MULTIPLY
    KEY_NUMPAD_SUBTRACT
    KEY_NUMPAD_ADD
    KEY_NUMPAD_ENTER
    KEY_NUMPAD_EQUAL
    KEY_LEFT_SHIFT
    KEY_LEFT_CONTROL
    KEY_LEFT_ALT
    KEY_LEFT_SUPER
    KEY_RIGHT_SHIFT
    KEY_RIGHT_CONTROL
    KEY_RIGHT_ALT
    KEY_RIGHT_SUPER
    KEY_MENU
    '''

    __members__: ClassVar[dict] = ...  # read-only
    KEY_0: ClassVar[KeyboardInput] = ...
    KEY_1: ClassVar[KeyboardInput] = ...
    KEY_2: ClassVar[KeyboardInput] = ...
    KEY_3: ClassVar[KeyboardInput] = ...
    KEY_4: ClassVar[KeyboardInput] = ...
    KEY_5: ClassVar[KeyboardInput] = ...
    KEY_6: ClassVar[KeyboardInput] = ...
    KEY_7: ClassVar[KeyboardInput] = ...
    KEY_8: ClassVar[KeyboardInput] = ...
    KEY_9: ClassVar[KeyboardInput] = ...
    KEY_A: ClassVar[KeyboardInput] = ...
    KEY_APOSTROPHE: ClassVar[KeyboardInput] = ...
    KEY_B: ClassVar[KeyboardInput] = ...
    KEY_BACKSLASH: ClassVar[KeyboardInput] = ...
    KEY_BACKSPACE: ClassVar[KeyboardInput] = ...
    KEY_C: ClassVar[KeyboardInput] = ...
    KEY_CAPS_LOCK: ClassVar[KeyboardInput] = ...
    KEY_COMMA: ClassVar[KeyboardInput] = ...
    KEY_D: ClassVar[KeyboardInput] = ...
    KEY_DEL: ClassVar[KeyboardInput] = ...
    KEY_DOWN: ClassVar[KeyboardInput] = ...
    KEY_E: ClassVar[KeyboardInput] = ...
    KEY_END: ClassVar[KeyboardInput] = ...
    KEY_ENTER: ClassVar[KeyboardInput] = ...
    KEY_EQUAL: ClassVar[KeyboardInput] = ...
    KEY_ESCAPE: ClassVar[KeyboardInput] = ...
    KEY_F: ClassVar[KeyboardInput] = ...
    KEY_F1: ClassVar[KeyboardInput] = ...
    KEY_F10: ClassVar[KeyboardInput] = ...
    KEY_F11: ClassVar[KeyboardInput] = ...
    KEY_F12: ClassVar[KeyboardInput] = ...
    KEY_F2: ClassVar[KeyboardInput] = ...
    KEY_F3: ClassVar[KeyboardInput] = ...
    KEY_F4: ClassVar[KeyboardInput] = ...
    KEY_F5: ClassVar[KeyboardInput] = ...
    KEY_F6: ClassVar[KeyboardInput] = ...
    KEY_F7: ClassVar[KeyboardInput] = ...
    KEY_F8: ClassVar[KeyboardInput] = ...
    KEY_F9: ClassVar[KeyboardInput] = ...
    KEY_G: ClassVar[KeyboardInput] = ...
    KEY_GRAVE_ACCENT: ClassVar[KeyboardInput] = ...
    KEY_H: ClassVar[KeyboardInput] = ...
    KEY_HOME: ClassVar[KeyboardInput] = ...
    KEY_I: ClassVar[KeyboardInput] = ...
    KEY_INSERT: ClassVar[KeyboardInput] = ...
    KEY_J: ClassVar[KeyboardInput] = ...
    KEY_K: ClassVar[KeyboardInput] = ...
    KEY_L: ClassVar[KeyboardInput] = ...
    KEY_LEFT: ClassVar[KeyboardInput] = ...
    KEY_LEFT_ALT: ClassVar[KeyboardInput] = ...
    KEY_LEFT_BRACKET: ClassVar[KeyboardInput] = ...
    KEY_LEFT_CONTROL: ClassVar[KeyboardInput] = ...
    KEY_LEFT_SHIFT: ClassVar[KeyboardInput] = ...
    KEY_LEFT_SUPER: ClassVar[KeyboardInput] = ...
    KEY_M: ClassVar[KeyboardInput] = ...
    KEY_MENU: ClassVar[KeyboardInput] = ...
    KEY_MINUS: ClassVar[KeyboardInput] = ...
    KEY_N: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_0: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_1: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_2: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_3: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_4: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_5: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_6: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_7: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_8: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_9: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_ADD: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_DEL: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_DIVIDE: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_ENTER: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_EQUAL: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_MULTIPLY: ClassVar[KeyboardInput] = ...
    KEY_NUMPAD_SUBTRACT: ClassVar[KeyboardInput] = ...
    KEY_NUM_LOCK: ClassVar[KeyboardInput] = ...
    KEY_O: ClassVar[KeyboardInput] = ...
    KEY_P: ClassVar[KeyboardInput] = ...
    KEY_PAGE_DOWN: ClassVar[KeyboardInput] = ...
    KEY_PAGE_UP: ClassVar[KeyboardInput] = ...
    KEY_PAUSE: ClassVar[KeyboardInput] = ...
    KEY_PERIOD: ClassVar[KeyboardInput] = ...
    KEY_PRINT_SCREEN: ClassVar[KeyboardInput] = ...
    KEY_Q: ClassVar[KeyboardInput] = ...
    KEY_R: ClassVar[KeyboardInput] = ...
    KEY_RIGHT: ClassVar[KeyboardInput] = ...
    KEY_RIGHT_ALT: ClassVar[KeyboardInput] = ...
    KEY_RIGHT_BRACKET: ClassVar[KeyboardInput] = ...
    KEY_RIGHT_CONTROL: ClassVar[KeyboardInput] = ...
    KEY_RIGHT_SHIFT: ClassVar[KeyboardInput] = ...
    KEY_RIGHT_SUPER: ClassVar[KeyboardInput] = ...
    KEY_S: ClassVar[KeyboardInput] = ...
    KEY_SCROLL_LOCK: ClassVar[KeyboardInput] = ...
    KEY_SEMICOLON: ClassVar[KeyboardInput] = ...
    KEY_SLASH: ClassVar[KeyboardInput] = ...
    KEY_SPACE: ClassVar[KeyboardInput] = ...
    KEY_T: ClassVar[KeyboardInput] = ...
    KEY_TAB: ClassVar[KeyboardInput] = ...
    KEY_U: ClassVar[KeyboardInput] = ...
    KEY_UP: ClassVar[KeyboardInput] = ...
    KEY_V: ClassVar[KeyboardInput] = ...
    KEY_W: ClassVar[KeyboardInput] = ...
    KEY_X: ClassVar[KeyboardInput] = ...
    KEY_Y: ClassVar[KeyboardInput] = ...
    KEY_Z: ClassVar[KeyboardInput] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class Mat33:
    ''' class isaacgym.gymapi.Mat33
    3x3 Matrix used for inetia tensor
    '''

    x: Vec3  # property x
    y: Vec3  # property y
    z: Vec3  # property z

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class Mat44:
    ''' class isaacgym.gymapi.Mat44
    4x4 Matrix
    '''

    w: Quat  # property w
    x: Quat  # property x
    y: Quat  # property y
    z: Quat  # property z

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class MeshNormalMode:
    __members__: ClassVar[dict] = ...  # read-only
    COMPUTE_PER_FACE: ClassVar[MeshNormalMode] = ...
    COMPUTE_PER_VERTEX: ClassVar[MeshNormalMode] = ...
    FROM_ASSET: ClassVar[MeshNormalMode] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class MeshType:
    ''' class isaacgym.gymapi.MeshType
    Types of mesh used by the simulator
    Members:
    
    MESH_NONE
    MESH_COLLISION : Collision mesh. Mesh is used only for Collision checks and calculations of inertia. For improved performance, it should be an approximation of the body volume by a coarse, convex mesh.
    MESH_VISUAL : Visual mesh. Mesh is used only for rendering purposes.
    MESH_VISUAL_AND_COLLISION : Visual and Collision Mesh. Mesh is used for both rendering and collision checks
    '''

    __members__: ClassVar[dict] = ...  # read-only
    MESH_COLLISION: ClassVar[MeshType] = ...
    MESH_NONE: ClassVar[MeshType] = ...
    MESH_VISUAL: ClassVar[MeshType] = ...
    MESH_VISUAL_AND_COLLISION: ClassVar[MeshType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class MouseInput:
    ''' class isaacgym.gymapi.MouseInput
    Members:
    MOUSE_LEFT_BUTTON
    MOUSE_RIGHT_BUTTON
    MOUSE_MIDDLE_BUTTON
    MOUSE_FORWARD_BUTTON
    MOUSE_BACK_BUTTON
    MOUSE_SCROLL_RIGHT
    MOUSE_SCROLL_LEFT
    MOUSE_SCROLL_UP
    MOUSE_SCROLL_DOWN
    MOUSE_MOVE_RIGHT
    MOUSE_MOVE_LEFT
    MOUSE_MOVE_UP
    MOUSE_MOVE_DOWN
    '''

    __members__: ClassVar[dict] = ...  # read-only
    MOUSE_BACK_BUTTON: ClassVar[MouseInput] = ...
    MOUSE_FORWARD_BUTTON: ClassVar[MouseInput] = ...
    MOUSE_LEFT_BUTTON: ClassVar[MouseInput] = ...
    MOUSE_MIDDLE_BUTTON: ClassVar[MouseInput] = ...
    MOUSE_MOVE_DOWN: ClassVar[MouseInput] = ...
    MOUSE_MOVE_LEFT: ClassVar[MouseInput] = ...
    MOUSE_MOVE_RIGHT: ClassVar[MouseInput] = ...
    MOUSE_MOVE_UP: ClassVar[MouseInput] = ...
    MOUSE_RIGHT_BUTTON: ClassVar[MouseInput] = ...
    MOUSE_SCROLL_DOWN: ClassVar[MouseInput] = ...
    MOUSE_SCROLL_LEFT: ClassVar[MouseInput] = ...
    MOUSE_SCROLL_RIGHT: ClassVar[MouseInput] = ...
    MOUSE_SCROLL_UP: ClassVar[MouseInput] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class OmniConnectionParams:
    enable_cache: bool
    enable_debug_output: bool
    password: str
    server: str
    username: str

    def __init__(self) -> None: ...


class ParticleState:
    dtype: ClassVar[dtype] = ...  # read-only
    normal: Quat
    position: Quat
    velocity: Vec3

    def __init__(self, *args, **kwargs) -> None: ...


class PerformanceTimers:
    ''' class isaacgym.gymapi.PerformanceTimers
    Amount of time in seconds spent doing the respective activity since last query
    '''

    frame_idling: float  # property frame_idling idling to keep updates close to graphics framerate
    graphics_image_retrieval: float  # property graphics_image_retrieval Copying images from the GPU to CPU
    graphics_sensor_rendering: float  # property graphics_sensor_rendering Rendering image sensors
    graphics_viewer_rendering: float  # property graphics_viewer_rendering Rendering the viewer
    physics_data_movement: float  # property physics_data_movement Copying physics state to/from the GPU
    physics_sim: float  # property physics_sim Running physics simulation
    total_time: float  # property total_time sum of all other timers

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class PhysXParams:
    ''' class isaacgym.gymapi.PhysXParams
    Simulation parameters used for PhysX physics engine
    '''

    always_use_articulations: bool  # property always_use_articulations If set, even single-body actors will be created as articulations
    bounce_threshold_velocity: float  # property bounce_threshold_velocity A contact with a relative velocity below this will not bounce. A typical value for simulation stability is about 2*gravity*dt/num_substeps.
    contact_collection: ContactCollection  # property contact_collection Contact collection mode
    contact_offset: float  # property contact_offset Shapes whose distance is less than the sum of their contactOffset values will generate contacts.
    default_buffer_size_multiplier: float  # property default_buffer_size_multiplier Default buffer size multiplier
    friction_correlation_distance: float  # property friction_correlation_distance Friction correlation distance
    friction_offset_threshold: float  # property friction_offset_threshold Friction offset threshold
    max_depenetration_velocity: float  # property max_depenetration_velocity The maximum velocity permitted to be introduced by the solver to correct for penetrations in contacts.
    max_gpu_contact_pairs: int  # property max_gpu_contact_pairs Maximum number of contact pairs
    num_position_iterations: int  # property num_position_iterations PhysX solver position iterations count. Range [1,255]
    num_subscenes: int  # property num_subscenes Number of subscenes for multithreaded simulation
    num_threads: int  # property num_threads Number of CPU threads used by PhysX. Should be set before the simulation is created. Setting this to 0 will run the simulation on the thread that calls PxScene::simulate(). A value greater than 0 will spawn numCores-1 worker threads.
    num_velocity_iterations: int  # property num_velocity_iterations PhysX solver velocity iterations count. Range [1,255]
    rest_offset: float  # property rest_offset Two shapes will come to rest at a distance equal to the sum of their restOffset values.
    solver_type: int  # property solver_type Type of solver used.  0 : PGS (Iterative sequential impulse solver 1 : TGS (Non-linear iterative solver, more robust but slightly more expensive
    use_gpu: bool  # property use_gpu Use PhysX GPU. Disabled at the moment.

    def __init__(self) -> None: ...


class PlaneParams:
    ''' class isaacgym.gymapi.PlaneParams
    Parameters for global ground plane
    '''

    distance: float  # property distance Ground plane distance from origin
    dynamic_friction: float  # property dynamic_friction Coefficient of dynamic friction
    normal: Vec3  # property normal Ground plane normal coefficient
    restitution: float  # property restitution Coefficient of restitution
    segmentation_id: int  # property segmentation_id SegmentationID value for segmentation ground truth
    static_friction: float  # property static_friction Coefficient of static friction

    def __init__(self) -> None: ...


class Quat:
    ''' class isaacgym.gymapi.Quat
    Quaternion representation in Gym
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('w', '<f4')])
    w: float  # property w
    x: float  # property x
    y: float  # property y
    z: float  # property z

    def __init__(self, x: float = ..., y: float = ..., z: float = ..., w: float = ...) -> None: ...

    def from_axis_angle(self, *args, **kwargs) -> Any: ...
    ''' static from_axis_angle(arg0: Vec3, arg1: float) -> Quat
    '''


    def from_buffer(self, *args, **kwargs) -> Any: ...
    ''' static from_buffer(arg0: buffer) -> object
    '''


    def from_euler_zyx(self, *args, **kwargs) -> Any: ...
    ''' static from_euler_zyx(arg0: float, arg1: float, arg2: float) -> Quat
    '''


    def inverse(self) -> Quat: ...
    ''' inverse(self: Quat) -> Quat
    '''


    def normalize(self) -> Quat: ...
    ''' normalize(self: Quat) -> Quat
    '''


    def rotate(self, arg0: Vec3) -> Vec3: ...
    ''' rotate(self: Quat, arg0: Vec3) -> Vec3
    '''


    def to_euler_zyx(self) -> Tuple[float, float, float]: ...
    ''' to_euler_zyx(self: Quat) -> Tuple[float, float, float]
    '''


    def __getstate__(self) -> tuple: ...

    def __mul__(self, arg0: Quat) -> Quat: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class RigidBodyProperties:
    ''' class isaacgym.gymapi.RigidBodyProperties
    Set of properties used for rigid bodies.
    '''

    com: Vec3  # property com center of mass in body space
    flags: int  # property flags Flags to enable certain behaivors on Rigid Bodies simulation. See isaacgym.gymapi.BodyFlags
    inertia: Mat33  # property inertia Inertia tensor relative to the center of mass.
    invInertia: Mat33  # property invInertia Inverse of Inertia tensor.
    invMass: float  # property invMass Inverse of mass value.
    mass: float  # property mass mass value, in kg

    def __init__(self) -> None: ...


class RigidBodyState:
    ''' class isaacgym.gymapi.RigidBodyState
    Containing states to get/set for a rigid body in the simulation
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('pose', [('p', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]), ('r', [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('w', '<f4')])]), ('vel', [('linear', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]), ('angular', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')])])])
    pose: Transform  # property pose Transform with position and orientation of rigid body
    vel: Any  # property vel Set of angular and linear velocities of rigid body

    def __init__(self, *args, **kwargs) -> None: ...


class RigidContact:
    ''' class isaacgym.gymapi.RigidContact
    Rigid Bodies contact information. Each contact in simulation generates a set of information.
    '''

    def __init__(self) -> None: ...

    @property
    def body0(self) -> int: ...
    ''' property body0
    Colliding rigid body indexes in the environment, -1 if it is ground plane
    '''


    @property
    def body1(self) -> int: ...
    ''' property body1
    Colliding rigid body indexes in the environment, -1 if it is ground plane
    '''


    @property
    def env0(self) -> int: ...
    ''' property env0
    Environment contact body0 belongs to, -1 if it is shared/unrecognized env
    '''


    @property
    def env1(self) -> int: ...
    ''' property env1
    Environment contact body1 belongs to, -1 if it is shared/unrecognized env
    '''


    @property
    def friction(self) -> float: ...
    ''' property friction
    Effective coefficient of Friction between bodies pair
    '''


    @property
    def initial_overlap(self) -> float: ...
    ''' property initial_overlap
    Amount of overlap along normal direction at the start of the time-step
    '''


    @property
    def lambda(self) -> float: ...
    ''' property lambda
    Contact force magnitude
    '''


    @property
    def lambda_friction(self) -> Vec2: ...
    ''' property lambda_friction
    Friction forces magnitudes. The direction of the friction force is the projection on the normal plane of the relative velocity of the bodies.
    '''


    @property
    def local_pos0(self) -> Vec3: ...
    ''' property local_pos0
    Local space position of the body0 contact feature excluding thickness, normal forces applied here
    '''


    @property
    def local_pos1(self) -> Vec3: ...
    ''' property local_pos1
    Local space position of the body1 contact feature excluding thickness, normal forces applied here
    '''


    @property
    def min_dist(self) -> float: ...
    ''' property min_dist
    Minimum distance to try and maintain along the contact normal between the two points
    '''


    @property
    def normal(self) -> Vec3: ...
    ''' property normal
    Contact normal from body0->body1 in world space
    '''


    @property
    def offset0(self) -> Vec3: ...
    ''' property offset0
    The local space offset from the feature localPos0 to the surface. That's the location where friction will be applied
    '''


    @property
    def offset1(self) -> Vec3: ...
    ''' property offset1
    The local space offset from the feature localPos1 to the surface. That's the location where friction will be applied
    '''


    @property
    def rolling_friction(self) -> float: ...
    ''' property rolling_friction
    Effective coeffitienc of Rolling Friction between bodies pair
    '''


    @property
    def torsion_friction(self) -> float: ...
    ''' property torsion_friction
    Effective coefficient of Torsional friction between bodies pair
    '''



class RigidShapeProperties:
    ''' class isaacgym.gymapi.RigidShapeProperties
    Set of properties used for all rigid shapes.
    '''

    compliance: float  # property compliance Coefficient of compliance. Determines how compliant the shape is. The smaller the value, the stronger the material will hold its shape. Value should be greater or equal to zero.
    contact_offset: float  # property contact_offset Distance at which contacts are generated (used with PhysX only
    filter: int  # property filter Collision filter bitmask - shapes A and B only collide if (filterA & filterB) == 0.
    friction: float  # property friction Coefficient of static friction. Value should be equal or greater than zero.
    rest_offset: float  # property rest_offset How far objects should come to rest from the surface of this body (used with PhysX only
    restitution: float  # property restitution Coefficient of restitution. It's the ratio of the final to initial velocity after the rigid body collides. Range [0,1]
    rolling_friction: float  # property rolling_friction Coefficient of rolling friction.
    thickness: float  # property thickness How far objects should come to rest from the surface of this body (used with Flex only)
    torsion_friction: float  # property torsion_friction Coefficient of torsion friction.

    def __init__(self) -> None: ...


class Sim:
    def __init__(self, *args, **kwargs) -> None: ...


class SimParams:
    ''' class isaacgym.gymapi.SimParams
    Gym Simulation Parameters
    '''

    dt: float  # property dt Simulation step size
    enable_actor_creation_warning: bool  # property enable_actor_creation_warning
    flex: FlexParams  # property flex Flex specific simulation parameters (See isaacgym.gymapi.FlexParams)
    gravity: Vec3  # property gravity 3-Dimension vector representing gravity force in Newtons.
    num_client_threads: int  # property num_client_threads
    physx: PhysXParams  # property physx PhysX specific simulation parameters (See isaacgym.gymapi.PhysXParams)
    stress_visualization: bool  # property stress_visualization
    stress_visualization_max: float  # property stress_visualization_max
    stress_visualization_min: float  # property stress_visualization_min
    substeps: int  # property substeps Number of subSteps for simulation
    up_axis: UpAxis  # property up_axis Up axis
    use_gpu_pipeline: bool  # property use_gpu_pipeline

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class SimType:
    ''' class isaacgym.gymapi.SimType
    Simulation Backend type
    Members:
    
    SIM_PHYSX : PhysX Backend
    SIM_FLEX : Flex Backend
    '''

    __members__: ClassVar[dict] = ...  # read-only
    SIM_FLEX: ClassVar[SimType] = ...
    SIM_PHYSX: ClassVar[SimType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class SoftContact:
    def __init__(self) -> None: ...

    @property
    def bodyIndex(self) -> int: ...

    @property
    def env0(self) -> int: ...

    @property
    def env1(self) -> int: ...

    @property
    def lambda(self) -> float: ...

    @property
    def normal(self) -> Vec3: ...

    @property
    def particleBarys(self) -> Vec3: ...

    @property
    def particleIndices(self) -> Int3: ...


class SoftMaterial:
    ''' class isaacgym.gymapi.SoftMaterial
    Soft Material definition
    '''

    activation: float  # property activation Current fiber activation.
    activationMax: float  # property activationMax Maximum activation value.
    damping: float  # property damping Material damping.
    model: SoftMaterialType  # property model Model type, See isaacgym.gymapi.SoftMaterialType
    poissons: float  # property poissons Poisson Ration.
    youngs: float  # property youngs Young Modulus.

    def __init__(self) -> None: ...


class SoftMaterialType:
    ''' class isaacgym.gymapi.SoftMaterialType
    Types of soft material supported in simulation
    Members:
    
    MAT_COROTATIONAL : Lagrange co-rotational formulation of finite elements
    MAT_NEOHOOKEAN : Neo-Hookean formulation of finite elements
    '''

    __members__: ClassVar[dict] = ...  # read-only
    MAT_COROTATIONAL: ClassVar[SoftMaterialType] = ...
    MAT_NEOHOOKEAN: ClassVar[SoftMaterialType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class SpatialForce:
    force: Vec3
    torque: Vec3

    def __init__(self) -> None: ...


class TendonProperties:
    damping: float
    fixed_lower_limit: float
    fixed_spring_rest_length: float
    fixed_upper_limit: float
    is_fixed_limited: bool
    limit_stiffness: float
    stiffness: float

    def __init__(self) -> None: ...

    @property
    def num_attachments(self) -> int: ...

    @property
    def num_joints(self) -> int: ...

    @property
    def type(self) -> TendonType: ...


class TendonType:
    ''' class isaacgym.gymapi.TendonType
    Tendon type
    Members:
    
    TENDON_FIXED : Fixed tendon
    TENDON_SPATIAL : Spatial tendon
    '''

    __members__: ClassVar[dict] = ...  # read-only
    TENDON_FIXED: ClassVar[TendonType] = ...
    TENDON_SPATIAL: ClassVar[TendonType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class Tensor:
    ''' class isaacgym.gymapi.Tensor
    Internal wrapper class of tensors.
    '''

    data_address: int  # property data_address address of data
    device: int  # property device
    dtype: Any  # property dtype data type
    own_data: bool  # property own_data flag for ownership
    shape: tuple  # property shape tensor shape

    def __init__(self) -> None: ...

    @property
    def data_ptr(self) -> capsule: ...
    ''' property data_ptr
    pointer to buffer
    '''


    @property
    def ndim(self) -> int: ...
    ''' property ndim
    number of dimensions
    '''



class TensorDataType:
    ''' class isaacgym.gymapi.TensorDataType
    Defines the data type of tensors.
    Members:
    
    DTYPE_FLOAT32 : float32
    DTYPE_UINT32 : uint32
    DTYPE_UINT64 : uint64
    DTYPE_UINT8 : uint8
    DTYPE_INT16 : int16
    '''

    __members__: ClassVar[dict] = ...  # read-only
    DTYPE_FLOAT32: ClassVar[TensorDataType] = ...
    DTYPE_INT16: ClassVar[TensorDataType] = ...
    DTYPE_UINT32: ClassVar[TensorDataType] = ...
    DTYPE_UINT64: ClassVar[TensorDataType] = ...
    DTYPE_UINT8: ClassVar[TensorDataType] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class Transform:
    ''' class isaacgym.gymapi.Transform
    Represents a transform in the system
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('p', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]), ('r', [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('w', '<f4')])])
    p: Vec3  # property p Position, in meters
    r: Quat  # property r Rotation Quaternion, represented in the format \(x\hat{i} + y\hat{j} + z\hat{k} + w\)

    def __init__(self, p: Vec3 = ..., r: Quat = ...) -> None: ...

    def from_buffer(self, *args, **kwargs) -> Any: ...
    ''' static from_buffer(arg0: buffer) -> object
    '''


    def inverse(self) -> Transform: ...
    ''' inverse(self: Transform) -> Transform
    Returns:
    the inverse of this transform.
    
    Return type:
    isaacgym.gymapi.Transform
    '''


    def transform_point(self, arg0: Vec3) -> Vec3: ...
    ''' transform_point(self: Transform, arg0: Vec3) -> Vec3
    Rotates point by transform quatertnion and adds transform offset
    Parameters:
    param1 (isaacgym.gymapi.Vec3) - Point to transform.
    
    Returns:
    The transformed point.
    
    Return type:
    isaacgym.gymapi.Vec3
    '''


    def transform_points(self, arg0: numpy.ndarray[Vec3]) -> numpy.ndarray[Vec3]: ...
    ''' transform_points(self: Transform, arg0: numpy.ndarray[Vec3]) -> numpy.ndarray[Vec3]
    Rotates points by transform quatertnion and adds transform offset
    Parameters:
    param1 (numpy.ndarray of isaacgym.gymapi.Vec3) - Points to transform.
    
    Returns:
    The transformed points.
    
    Return type:
    numpy.ndarray[isaacgym.gymapi.Vec3]
    '''


    def transform_vector(self, arg0: Vec3) -> Vec3: ...
    ''' transform_vector(self: Transform, arg0: Vec3) -> Vec3
    Rotates vector by transform quatertnion
    Parameters:
    param1 (isaacgym.gymapi.Vec3) - Vector to transform.
    
    Returns:
    The transformed vector.
    
    Return type:
    isaacgym.gymapi.Vec3
    '''


    def transform_vectors(self, arg0: numpy.ndarray[Vec3]) -> numpy.ndarray[Vec3]: ...
    ''' transform_vectors(self: Transform, arg0: numpy.ndarray[Vec3]) -> numpy.ndarray[Vec3]
    Rotates vectors by transform quatertnion
    Parameters:
    param1 (numpy.ndarray of isaacgym.gymapi.Vec3) - Vectors to transform.
    
    Returns:
    The transformed vectors.
    
    Return type:
    numpy.ndarray[isaacgym.gymapi.Vec3]
    '''


    def __getstate__(self) -> tuple: ...

    def __mul__(self, arg0: Transform) -> Transform: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class TriangleMeshParams:
    ''' class isaacgym.gymapi.TriangleMeshParams
    Triangle Mesh properties
    '''

    dynamic_friction: float  # property dynamic_friction Coefficient of dynamic friction
    nb_triangles: int  # property nb_triangles Number of triangles
    nb_vertices: int  # property nb_vertices Number of vertices
    restitution: float  # property restitution Coefficient of restitution
    segmentation_id: int  # property segmentation_id SegmentationID value for segmentation ground truth
    static_friction: float  # property static_friction Coefficient of static friction
    transform: Transform  # property transform Transform to apply to heightfield

    def __init__(self) -> None: ...


class UpAxis:
    ''' class isaacgym.gymapi.UpAxis
    Up axis
    Members:
    
    UP_AXIS_Y : Y axis points up
    UP_AXIS_Z : Z axis points up
    '''

    __members__: ClassVar[dict] = ...  # read-only
    UP_AXIS_Y: ClassVar[UpAxis] = ...
    UP_AXIS_Z: ClassVar[UpAxis] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __and__(self, other) -> Any: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    def __xor__(self, other) -> Any: ...

    @property
    def name(self) -> str: ...


class UsdExportOptions:
    export_physics: bool
    material_mode: UsdMaterialMode
    scale: float
    single_file: bool
    use_geom_subsets: bool
    use_physics_visuals: bool

    def __init__(self) -> None: ...

    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class UsdExporter:
    def __init__(self, *args, **kwargs) -> None: ...


class UsdMaterialMode:
    __members__: ClassVar[dict] = ...  # read-only
    USD_MATERIAL_DISPLAY_COLOR: ClassVar[UsdMaterialMode] = ...
    USD_MATERIAL_MDL: ClassVar[UsdMaterialMode] = ...
    USD_MATERIAL_PREVIEW_SURFACE: ClassVar[UsdMaterialMode] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, arg0: int) -> None: ...

    def __eq__(self, other) -> Any: ...

    def __ge__(self, other) -> Any: ...

    def __getstate__(self) -> Any: ...

    def __gt__(self, other) -> Any: ...

    def __hash__(self) -> Any: ...

    def __int__(self) -> int: ...

    def __le__(self, other) -> Any: ...

    def __lt__(self, other) -> Any: ...

    def __ne__(self, other) -> Any: ...

    def __setstate__(self, state) -> Any: ...

    @property
    def name(self) -> str: ...


class Vec2:
    dtype: ClassVar[dtype] = ...  # read-only
    x: float
    y: float

    def __init__(self, x: float = ..., y: float = ...) -> None: ...

    def from_buffer(self, *args, **kwargs) -> Any: ...

    def __add__(self, arg0: Vec2) -> Vec2: ...

    def __getstate__(self) -> tuple: ...

    def __mul__(self, arg0: float) -> Vec2: ...

    def __neg__(self) -> Vec2: ...

    def __setstate__(self, arg0: tuple) -> None: ...

    def __sub__(self, arg0: Vec2) -> Vec2: ...

    def __truediv__(self, arg0: float) -> Vec2: ...


class Vec3:
    ''' class isaacgym.gymapi.Vec3
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    x: float  # property x
    y: float  # property y
    z: float  # property z

    def __init__(self, x: float = ..., y: float = ..., z: float = ...) -> None: ...

    def cross(self, arg0: Vec3) -> Vec3: ...
    ''' cross(self: Vec3, arg0: Vec3) -> Vec3
    '''


    def dot(self, arg0: Vec3) -> float: ...
    ''' dot(self: Vec3, arg0: Vec3) -> float
    '''


    def from_buffer(self, *args, **kwargs) -> Any: ...
    ''' static from_buffer(arg0: buffer) -> object
    '''


    def length(self) -> float: ...
    ''' length(self: Vec3) -> float
    '''


    def length_sq(self) -> float: ...
    ''' length_sq(self: Vec3) -> float
    '''


    def normalize(self) -> Vec3: ...
    ''' normalize(self: Vec3) -> Vec3
    '''


    def __add__(self, arg0: Vec3) -> Vec3: ...

    def __getstate__(self) -> tuple: ...

    def __mul__(self, arg0: float) -> Vec3: ...

    def __neg__(self) -> Vec3: ...

    def __setstate__(self, arg0: tuple) -> None: ...

    def __sub__(self, arg0: Vec3) -> Vec3: ...

    def __truediv__(self, arg0: float) -> Vec3: ...


class Velocity:
    ''' class isaacgym.gymapi.Velocity
    Holds linear and angular velocities, in $m/s$ and $radians/s$
    '''

    dtype: ClassVar[dtype] = ...  # read-only  # dtype = dtype([('linear', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]), ('angular', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')])])
    angular: Vec3  # property angular angular velocity component
    linear: Vec3  # property linear Linear velocity component

    def __init__(self, p: Vec3 = ..., q: Vec3 = ...) -> None: ...

    def from_buffer(self, *args, **kwargs) -> Any: ...
    ''' static from_buffer(arg0: buffer) -> object
    '''


    def __getstate__(self) -> tuple: ...

    def __setstate__(self, arg0: tuple) -> None: ...


class Version:
    ''' class isaacgym.gymapi.Version
    Defines a major and minor version
    '''

    major: int  # property major
    minor: int  # property minor

    def __init__(self) -> None: ...


class VhacdParams:
    ''' class isaacgym.gymapi.VhacdParams
    VHACD Convex Decomposition parameters
    '''

    alpha: float  # property alpha Controls the bias toward clipping along symmetry planes.  0.0-1.0.  Default 0.05.
    beta: float  # property beta Controls the bias toward clipping along revolution axes.  0.0-1.0.  Default 0.05.
    concavity: float  # property concavity Maximum concavity.  0.0-1.0.  Default 0.0.
    convex_hull_approximation: bool  # property convex_hull_approximation Default True.
    convex_hull_downsampling: int  # property convex_hull_downsampling Controls the precision of the convex-hull generation process during the clipping plane selection stage.  1-16.  Default 4.
    max_convex_hulls: int  # property max_convex_hulls Maximum number of convex hulls.  Default 64.
    max_num_vertices_per_ch: int  # property max_num_vertices_per_ch Controls the maximum number of vertices per convex-hull.  4-1024.  Default 64.
    min_volume_per_ch: float  # property min_volume_per_ch Controls the adaptive sampling of the generated convex-hulls.  0.0-0.01.  Default 0.0001.
    mode: 0  # property mode tetrahedron-based approximate convex decomposition.  Default 0. Type: 0  Type: voxel-based approximate convex decomposition, 1
    ocl_acceleration: bool  # property ocl_acceleration Default True.
    pca: int  # property pca Enable/disable normalizing the mesh before applying the convex decomposition.  0-1.  Default 0.
    plane_downsampling: int  # property plane_downsampling Controls the granularity of the search for the “best” clipping plane.  1-16.  Default 4.
    project_hull_vertices: bool  # property project_hull_vertices Default True.
    resolution: int  # property resolution Maximum number of voxels generated during the voxelization stage.  10,000-64,000,000.  Default 100,000.

    def __init__(self) -> None: ...


class Viewer:
    def __init__(self, *args, **kwargs) -> None: ...


def acquire_gym(*args, **kwargs) -> Gym: ...


def carb_init(config_str: str = ...) -> bool: ...


def cross(*args, **kwargs) -> Any: ...


@overload
def dot(arg0, arg1) -> numpy.ndarray[numpy.float32]: ...


@overload
def dot(arg0, arg1) -> numpy.ndarray[numpy.float32]: ...


def eulers_to_quats_zyx(*args, **kwargs) -> Any: ...


def quats_to_eulers_zyx(*args, **kwargs) -> Any: ...


def rotate(*args, **kwargs) -> Any: ...


def rotate_inverse(*args, **kwargs) -> Any: ...


def transform(*args, **kwargs) -> Any: ...


def transform_inverse(*args, **kwargs) -> Any: ...
