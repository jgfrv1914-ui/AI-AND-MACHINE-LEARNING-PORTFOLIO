/* Hero 3D avatar: a procedural robot waving its hand.
   Built from Three.js primitives (no external mesh) so it stays small and uses
   exactly the palette defined in styles.css. */
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const mount = document.getElementById("hero-robot");
if (mount) initRobot(mount);

function initRobot(mount) {
  const VIOLET = 0x9b7cff, CYAN = 0x62e6e5, LIME = 0xd4f07b;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(0, 0.35, 6.4);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  mount.appendChild(renderer.domElement);

  // --- Materials ------------------------------------------------------------
  const shell = new THREE.MeshStandardMaterial({ color: 0x20213a, metalness: 0.55, roughness: 0.38 });
  const plate = new THREE.MeshStandardMaterial({ color: 0x2c2d4d, metalness: 0.7, roughness: 0.3 });
  const joint = new THREE.MeshStandardMaterial({ color: 0x14151f, metalness: 0.85, roughness: 0.45 });
  const glowViolet = new THREE.MeshStandardMaterial({ color: VIOLET, emissive: VIOLET, emissiveIntensity: 1.1, roughness: 0.4 });
  const glowCyan = new THREE.MeshStandardMaterial({ color: CYAN, emissive: CYAN, emissiveIntensity: 1.3, roughness: 0.35 });
  const glowLime = new THREE.MeshStandardMaterial({ color: LIME, emissive: LIME, emissiveIntensity: 1.5, roughness: 0.4 });

  const box = (w, h, d, r = 0.06) => new RoundedBoxGeometry(w, h, d, 4, r);
  const add = (parent, geo, mat, x = 0, y = 0, z = 0) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    parent.add(m);
    return m;
  };

  // --- Assembly -------------------------------------------------------------
  const robot = new THREE.Group();
  scene.add(robot);

  // Torso
  const torso = add(robot, box(1.28, 1.42, 0.78, 0.22), shell, 0, 0.18);
  add(torso, box(0.86, 0.5, 0.06, 0.1), plate, 0, 0.3, 0.4);
  const core = add(torso, new THREE.TorusGeometry(0.19, 0.055, 16, 40), glowCyan, 0, -0.16, 0.4);
  add(torso, new THREE.SphereGeometry(0.1, 24, 24), glowCyan, 0, -0.16, 0.42);
  // Shoulder pads
  add(robot, new THREE.SphereGeometry(0.26, 24, 24), plate, -0.74, 0.62);
  add(robot, new THREE.SphereGeometry(0.26, 24, 24), plate, 0.74, 0.62);

  // Neck + head (its own group: it tracks the cursor)
  add(robot, new THREE.CylinderGeometry(0.15, 0.17, 0.2, 20), joint, 0, 0.98);
  const head = new THREE.Group();
  head.position.set(0, 1.42, 0);
  robot.add(head);
  add(head, box(1.08, 0.86, 0.8, 0.26), shell);
  add(head, box(0.82, 0.36, 0.08, 0.15), joint, 0, 0.04, 0.4); // visor
  const eyeL = add(head, new THREE.CapsuleGeometry(0.075, 0.1, 6, 16), glowCyan, -0.19, 0.04, 0.45);
  const eyeR = add(head, new THREE.CapsuleGeometry(0.075, 0.1, 6, 16), glowCyan, 0.19, 0.04, 0.45);
  eyeL.rotation.z = eyeR.rotation.z = Math.PI / 2;
  // Ear pieces
  add(head, new THREE.CylinderGeometry(0.13, 0.13, 0.1, 20), glowViolet, -0.56, -0.02, 0).rotation.z = Math.PI / 2;
  add(head, new THREE.CylinderGeometry(0.13, 0.13, 0.1, 20), glowViolet, 0.56, -0.02, 0).rotation.z = Math.PI / 2;
  // Antenna
  add(head, new THREE.CylinderGeometry(0.025, 0.025, 0.42, 12), joint, 0.26, 0.62);
  const antenna = add(head, new THREE.SphereGeometry(0.085, 20, 20), glowLime, 0.26, 0.86);

  // Arm: shoulder -> forearm -> open hand
  function buildArm(side) {
    const shoulder = new THREE.Group();
    shoulder.position.set(0.72 * side, 0.6, 0);
    robot.add(shoulder);
    add(shoulder, new THREE.CapsuleGeometry(0.15, 0.42, 8, 20), shell, 0, -0.34);

    const elbow = new THREE.Group();
    elbow.position.set(0, -0.68, 0);
    shoulder.add(elbow);
    add(elbow, new THREE.SphereGeometry(0.15, 20, 20), joint);
    add(elbow, new THREE.CapsuleGeometry(0.13, 0.4, 8, 20), plate, 0, -0.32);

    const hand = new THREE.Group();
    hand.position.set(0, -0.62, 0);
    elbow.add(hand);
    add(hand, box(0.3, 0.3, 0.16, 0.08), shell, 0, -0.1);
    // Four open fingers + thumb: the wave reads better with an open hand.
    for (let i = 0; i < 4; i++) {
      add(hand, box(0.055, 0.26, 0.09, 0.03), plate, -0.1 + i * 0.067, -0.35);
    }
    add(hand, box(0.055, 0.19, 0.09, 0.03), plate, -0.19, -0.14).rotation.z = 0.6 * -side;
    return { shoulder, elbow, hand };
  }

  const armWave = buildArm(-1); // viewer-left arm is the one that waves
  const armRest = buildArm(1);
  armRest.shoulder.rotation.z = 0.14;
  armRest.elbow.rotation.z = -0.22;

  // Floating ring beneath the robot
  const ring = add(robot, new THREE.TorusGeometry(1.05, 0.022, 12, 72), glowViolet, 0, -1.12);
  ring.rotation.x = Math.PI / 2;

  // --- Lights -----------------------------------------------------------------
  scene.add(new THREE.AmbientLight(0x6f78a8, 0.85));
  const key = new THREE.DirectionalLight(VIOLET, 2.4);
  key.position.set(3, 4, 4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(CYAN, 2.1);
  rim.position.set(-4, 1.5, -2.5);
  scene.add(rim);
  const fill = new THREE.PointLight(LIME, 6, 9);
  fill.position.set(-1.6, -1.4, 2.6);
  scene.add(fill);

  // --- Interaction ------------------------------------------------------------
  const pointer = { x: 0, y: 0 };
  addEventListener("pointermove", (e) => {
    pointer.x = (e.clientX / innerWidth) * 2 - 1;
    pointer.y = (e.clientY / innerHeight) * 2 - 1;
  }, { passive: true });

  function resize() {
    const { clientWidth: w, clientHeight: h } = mount;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(mount);
  resize();

  // Pause the loop while the hero is off screen.
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(mount);

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    if (!visible) return;
    const t = clock.getElapsedTime();

    // Idle float and sway
    robot.position.y = Math.sin(t * 1.3) * 0.075;
    robot.rotation.y = -0.12 + Math.sin(t * 0.55) * 0.14 + pointer.x * 0.28;
    robot.rotation.z = Math.sin(t * 0.9) * 0.018;

    // The head tracks the cursor with slightly more travel than the body.
    head.rotation.y = pointer.x * 0.32;
    head.rotation.x = pointer.y * 0.2 + Math.sin(t * 1.3) * 0.02;

    // Wave: arm raised, forearm swinging.
    armWave.shoulder.rotation.z = -2.15 + Math.sin(t * 1.6) * 0.05;
    armWave.elbow.rotation.z = -0.99 + Math.sin(t * 3.6) * 0.46;
    armWave.hand.rotation.z = Math.sin(t * 3.6 + 0.6) * 0.22;

    // Resting arm: a faint breathing motion.
    armRest.shoulder.rotation.z = 0.14 + Math.sin(t * 1.3) * 0.045;

    // Accent pulses
    const pulse = 0.85 + Math.sin(t * 2.4) * 0.35;
    core.material.emissiveIntensity = 1.3 * pulse;
    antenna.material.emissiveIntensity = 1.5 * (0.6 + Math.abs(Math.sin(t * 1.8)) * 0.8);
    ring.rotation.z = t * 0.5;

    renderer.render(scene, camera);
  });
}
