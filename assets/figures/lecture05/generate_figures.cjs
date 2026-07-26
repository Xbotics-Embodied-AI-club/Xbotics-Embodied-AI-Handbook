const fs = require("node:fs");
const path = require("node:path");
const sharp = require("/Users/liuzhikai/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");

const OUT = path.join(__dirname, "original");
fs.mkdirSync(OUT, { recursive: true });

const W = 1600;
const H = 900;
const C = {
  navy: "#123B64",
  blue: "#1F6FEB",
  cyan: "#45B7D1",
  pale: "#EAF3FF",
  paler: "#F6FAFF",
  ink: "#172B3A",
  gray: "#5B6B7A",
  line: "#A9BED1",
  darkLine: "#55738E",
  white: "#FFFFFF",
  amber: "#E9A23B",
  red: "#C94B4B",
  green: "#2F8F6B",
};

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function base(title, subtitle = "") {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs>
    <marker id="arrow-blue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L12,6 L0,12 Z" fill="${C.blue}"/>
    </marker>
    <marker id="arrow-dark" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L12,6 L0,12 Z" fill="${C.darkLine}"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#123B64" flood-opacity="0.12"/>
    </filter>
    <style>
      text { font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: ${C.ink}; }
      .title { font-size: 42px; font-weight: 700; letter-spacing: 1px; }
      .subtitle { font-size: 21px; fill: ${C.gray}; }
      .h { font-size: 27px; font-weight: 700; }
      .m { font-size: 23px; font-weight: 600; }
      .b { font-size: 20px; }
      .s { font-size: 17px; fill: ${C.gray}; }
      .mono { font-family: "SFMono-Regular", Consolas, monospace; font-size: 18px; }
    </style>
  </defs>
  <rect width="${W}" height="${H}" fill="${C.white}"/>
  <rect x="0" y="0" width="18" height="${H}" fill="${C.blue}"/>
  <text x="70" y="72" class="title">${esc(title)}</text>
  ${subtitle ? `<text x="72" y="108" class="subtitle">${esc(subtitle)}</text>` : ""}
  <line x1="70" y1="130" x2="1530" y2="130" stroke="${C.line}" stroke-width="2"/>`;
}

function end() {
  return "</svg>";
}

function rounded(x, y, w, h, fill = C.paler, stroke = C.line, sw = 2, r = 24, shadow = false) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${shadow ? ' filter="url(#shadow)"' : ""}/>`;
}

function lines(x, y, textLines, cls = "b", lineH = 31, anchor = "start", fill = "") {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}"${fill ? ` style="fill:${fill}"` : ""}>${textLines
    .map((t, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : lineH}">${esc(t)}</tspan>`)
    .join("")}</text>`;
}

function arrow(x1, y1, x2, y2, color = C.blue, dashed = false, width = 4) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dashed ? ' stroke-dasharray="12 10"' : ""} marker-end="url(#arrow-${color === C.blue ? "blue" : "dark"})"/>`;
}

function pill(x, y, w, text, fill = C.pale, stroke = C.blue) {
  return `${rounded(x, y, w, 48, fill, stroke, 2, 24)}<text x="${x + w / 2}" y="${y + 32}" class="b" font-weight="700" text-anchor="middle">${esc(text)}</text>`;
}

function writeFigure(name, svg) {
  const svgPath = path.join(OUT, `${name}.svg`);
  const pngPath = path.join(OUT, `${name}.png`);
  fs.writeFileSync(svgPath, svg);
  return sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(pngPath);
}

function taskContract() {
  const items = [
    ["1", "目标", "机器人要完成什么"],
    ["2", "模型", "身体与执行器是什么"],
    ["3", "场景", "世界里有哪些对象"],
    ["4", "观测", "控制器能够读取什么"],
    ["5", "动作", "指令以什么口径下发"],
    ["6", "物理与控制", "世界怎样推进、控制怎样更新"],
    ["7", "判定", "成功、失败与超时怎样区分"],
  ];
  const positions = [
    [140, 195],
    [520, 165],
    [900, 195],
    [1115, 400],
    [900, 610],
    [455, 640],
    [120, 560],
  ];
  let s = base("七项任务契约", "先把真实任务写成可执行约定，再打开仿真器");
  s += `<circle cx="800" cy="445" r="155" fill="${C.navy}"/>`;
  s += lines(800, 420, ["可复盘的", "仿真实验"], "h", 43, "middle", C.white);
  s += `<text x="800" y="505" class="s" text-anchor="middle" style="fill:#DDEBFA">可控 · 可测 · 可重复 · 可解释</text>`;
  items.forEach((it, i) => {
    const [x, y] = positions[i];
    const w = i === 5 ? 360 : 300;
    s += rounded(x, y, w, 112, i % 2 ? C.white : C.paler, i % 2 ? C.darkLine : C.blue, 3, 22, true);
    s += `<circle cx="${x + 43}" cy="${y + 38}" r="23" fill="${i % 2 ? C.navy : C.blue}"/><text x="${x + 43}" y="${y + 46}" class="b" text-anchor="middle" style="fill:${C.white}" font-weight="700">${it[0]}</text>`;
    s += `<text x="${x + 80}" y="${y + 45}" class="m">${it[1]}</text>`;
    s += `<text x="${x + 28}" y="${y + 83}" class="s">${it[2]}</text>`;
    const bx = x + w / 2;
    const by = y + 56;
    const dx = 800 - bx;
    const dy = 445 - by;
    const norm = Math.hypot(dx, dy);
    s += arrow(bx + (dx / norm) * 62, by + (dy / norm) * 62, 800 - (dx / norm) * 174, 445 - (dy / norm) * 174, C.darkLine, true, 3);
  });
  s += end();
  return s;
}

function contractMap() {
  let s = base("任务契约落在哪里", "MJCF 描述世界与执行器，Python 负责闭环、时序和判定");
  s += pill(80, 155, 285, "七项任务契约");
  s += rounded(90, 225, 660, 575, C.paler, C.blue, 3, 26, true);
  s += rounded(850, 225, 660, 575, C.white, C.darkLine, 3, 26, true);
  s += `<text x="420" y="275" class="h" text-anchor="middle">MJCF / 机器人资产</text>`;
  s += `<text x="1180" y="275" class="h" text-anchor="middle">Python / 实验程序</text>`;
  const left = [
    ["模型", "body · joint · geom · inertial"],
    ["场景", "worldbody · freejoint · contact"],
    ["观测锚点", "site · sensor"],
    ["动作入口", "actuator · ctrlrange · forcerange"],
    ["物理基线", "option/timestep · damping · friction"],
  ];
  const right = [
    ["目标", "target_pos"],
    ["观测读取", "qpos · qvel · site_xpos"],
    ["控制闭环", "Jacobian → IK → data.ctrl"],
    ["时钟", "physics step · control period · render"],
    ["判定与证据", "success · timeout · log · seed"],
  ];
  left.forEach((it, i) => {
    const y = 315 + i * 94;
    s += rounded(135, y, 570, 70, C.white, C.line, 2, 15);
    s += `<text x="170" y="${y + 29}" class="m">${it[0]}</text>`;
    s += `<text x="170" y="${y + 55}" class="mono">${esc(it[1])}</text>`;
  });
  right.forEach((it, i) => {
    const y = 315 + i * 94;
    s += rounded(895, y, 570, 70, C.paler, C.line, 2, 15);
    s += `<text x="930" y="${y + 29}" class="m">${it[0]}</text>`;
    s += `<text x="930" y="${y + 55}" class="mono">${esc(it[1])}</text>`;
  });
  s += arrow(750, 510, 850, 510, C.blue, false, 5);
  s += `<text x="800" y="485" class="s" text-anchor="middle">编译后由 name / id 对齐</text>`;
  s += end();
  return s;
}

function reachLoop() {
  let s = base("SO-101 Reach 闭环", "目标位置不直接驱动电机：必须经过误差、Jacobian、IK 与执行器");
  const nodes = [
    [70, 330, 205, 150, "目标位置", ["x*", "世界坐标"]],
    [320, 330, 215, 150, "位置误差", ["e = x* − x", "每个控制周期"]],
    [580, 330, 265, 150, "阻尼最小二乘 IK", ["Δq = Jᵀ(JJᵀ+λ²I)⁻¹e", "关节增量限幅"]],
    [890, 330, 220, 150, "关节目标", ["qcmd = clip(q+Δq)", "写入 data.ctrl"]],
    [1155, 330, 205, 150, "物理推进", ["mj_step", "5 ms / step"]],
    [1400, 330, 150, 150, "新观测", ["q, q̇, x", "记录日志"]],
  ];
  nodes.forEach((n, i) => {
    const [x, y, w, h, title, body] = n;
    s += rounded(x, y, w, h, i === 2 ? C.pale : C.white, i === 2 ? C.blue : C.darkLine, 3, 22, true);
    s += `<text x="${x + w / 2}" y="${y + 48}" class="m" text-anchor="middle">${title}</text>`;
    s += lines(x + w / 2, y + 88, body, i === 2 ? "s" : "b", 28, "middle");
    if (i < nodes.length - 1) s += arrow(x + w + 12, y + 75, nodes[i + 1][0] - 12, y + 75);
  });
  s += `<path d="M1475 505 C1475 700, 420 720, 420 505" fill="none" stroke="${C.darkLine}" stroke-width="4" stroke-dasharray="13 9" marker-end="url(#arrow-dark)"/>`;
  s += `<text x="930" y="695" class="b" text-anchor="middle">反馈：新末端位置进入下一次误差计算</text>`;
  s += pill(230, 185, 250, "物理步长 5 ms");
  s += pill(675, 185, 250, "控制周期 20 ms");
  s += pill(1120, 185, 250, "成功：误差保持达标");
  s += end();
  return s;
}

function pickFailures() {
  let s = base("Pick-place：四种看似相近、机制不同的失败", "先识别症状，再定位契约断点；不要把“接触数量”当作稳定抓取");
  const cards = [
    [70, 190, "01", "抓空", ["双指未形成对向接触", "先查预抓取位姿与几何"]],
    [450, 190, "02", "抬升滑脱", ["接触后物体未随夹爪", "查摩擦、夹持与加速度"]],
    [830, 190, "03", "接触弹飞", ["闭合瞬间速度突增", "查闭合速度与接触参数"]],
    [1210, 190, "04", "放置滑动", ["释放后仍有平动速度", "查释放高度、速度与桌面"]],
  ];
  cards.forEach((c, i) => {
    const [x, y, no, title, body] = c;
    s += rounded(x, y, 320, 500, i % 2 ? C.white : C.paler, i % 2 ? C.darkLine : C.blue, 3, 26, true);
    s += `<circle cx="${x + 58}" cy="${y + 58}" r="34" fill="${i % 2 ? C.navy : C.blue}"/><text x="${x + 58}" y="${y + 67}" class="m" text-anchor="middle" style="fill:${C.white}">${no}</text>`;
    s += `<text x="${x + 110}" y="${y + 68}" class="h">${title}</text>`;
    const ox = x + 160;
    const oy = y + 250;
    s += `<rect x="${ox - 46}" y="${oy - 50}" width="92" height="92" rx="12" fill="${C.pale}" stroke="${C.navy}" stroke-width="4"/>`;
    s += `<rect x="${ox - 112}" y="${oy - 92}" width="38" height="150" rx="12" fill="${C.white}" stroke="${C.darkLine}" stroke-width="6"/>`;
    s += `<rect x="${ox + 74}" y="${oy - 92}" width="38" height="150" rx="12" fill="${C.white}" stroke="${C.darkLine}" stroke-width="6"/>`;
    if (i === 0) s += `<line x1="${ox - 68}" y1="${oy}" x2="${ox - 52}" y2="${oy}" stroke="${C.red}" stroke-width="7" stroke-dasharray="8 8"/><line x1="${ox + 52}" y1="${oy}" x2="${ox + 68}" y2="${oy}" stroke="${C.red}" stroke-width="7" stroke-dasharray="8 8"/>`;
    if (i === 1) s += arrow(ox, oy + 58, ox, oy + 135, C.darkLine, true, 5);
    if (i === 2) s += `<path d="M${ox - 5} ${oy - 65} l-28 -30 M${ox + 5} ${oy - 65} l28 -30 M${ox - 38} ${oy - 50} l-28 -5 M${ox + 38} ${oy - 50} l28 -5" stroke="${C.red}" stroke-width="7"/>`;
    if (i === 3) s += `<line x1="${ox - 115}" y1="${oy + 70}" x2="${ox + 120}" y2="${oy + 70}" stroke="${C.darkLine}" stroke-width="7"/><path d="M${ox + 15} ${oy + 15} C${ox + 50} ${oy + 10},${ox + 90} ${oy + 20},${ox + 112} ${oy + 50}" fill="none" stroke="${C.red}" stroke-width="6" marker-end="url(#arrow-dark)"/>`;
    s += lines(x + 32, y + 400, body, "b", 35);
  });
  s += pill(390, 750, 820, "可信抓取证据 = 双指身份 + 接触方向/力 + 夹爪状态 + 物体随动");
  s += end();
  return s;
}

function clocks() {
  let s = base("仿真里至少有四个时钟", "“控制频率变了”只有在说明是哪一个时钟后才有意义");
  const x0 = 235;
  const x1 = 1490;
  const yRows = [230, 375, 520, 665];
  const labels = [
    ["物理积分", "timestep = 5 ms", C.navy],
    ["控制更新", "每 4 个 physics step → 20 ms", C.blue],
    ["策略/环境", "decimation 可再拉长动作保持", C.cyan],
    ["渲染与录屏", "可低频同步，不改变物理基线", C.gray],
  ];
  labels.forEach((l, ri) => {
    const y = yRows[ri];
    s += `<text x="70" y="${y + 7}" class="m">${l[0]}</text><text x="70" y="${y + 36}" class="s">${l[1]}</text>`;
    s += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${C.line}" stroke-width="3"/>`;
    const step = [38, 152, 304, 456][ri];
    for (let x = x0; x <= x1; x += step) {
      s += `<line x1="${x}" y1="${y - 26}" x2="${x}" y2="${y + 26}" stroke="${l[2]}" stroke-width="${ri === 0 ? 3 : 5}"/>`;
      if (ri > 0 && x + step <= x1) s += `<rect x="${x + 5}" y="${y - 13}" width="${step - 10}" height="26" rx="9" fill="${l[2]}" opacity="${ri === 3 ? 0.35 : 0.18}"/>`;
    }
  });
  s += `<path d="M235 180 v600 M615 180 v600 M995 180 v600 M1375 180 v600" stroke="${C.darkLine}" stroke-width="2" stroke-dasharray="8 10" opacity="0.55"/>`;
  s += `<text x="805" y="830" class="b" text-anchor="middle">对比实验一次只改变一个时钟，其余基线固定并写入日志</text>`;
  s += end();
  return s;
}

function isaacFlow() {
  let s = base("Isaac Lab：把任务契约组织成环境协议", "机器人资产改变了，目标—观测—动作—判定的阅读方法不变");
  s += rounded(65, 175, 435, 565, C.paler, C.blue, 3, 25, true);
  s += `<text x="282" y="225" class="h" text-anchor="middle">场景与资产层</text>`;
  [["Scene", "地面、机器人、目标"], ["Asset / Articulation", "USD 资产与关节系统"], ["Sensor", "相机、接触、IMU"]].forEach((it, i) => {
    const y = 280 + i * 135;
    s += rounded(115, y, 335, 92, C.white, C.line, 2, 18);
    s += `<text x="282" y="${y + 37}" class="m" text-anchor="middle">${it[0]}</text>`;
    s += `<text x="282" y="${y + 67}" class="s" text-anchor="middle">${it[1]}</text>`;
  });
  s += rounded(585, 175, 950, 565, C.white, C.darkLine, 3, 25, true);
  s += `<text x="1060" y="225" class="h" text-anchor="middle">环境任务层</text>`;
  const top = [
    [635, 285, "Observation", "策略读什么"],
    [925, 285, "Action", "策略输出什么"],
    [1215, 285, "Command", "任务条件是什么"],
  ];
  const bottom = [
    [635, 520, "Reward", "训练时鼓励什么"],
    [925, 520, "Termination", "成功/失败何时结束"],
    [1215, 520, "Metrics", "评测怎样统计"],
  ];
  [...top, ...bottom].forEach((it, i) => {
    s += rounded(it[0], it[1], 260, 125, i < 3 ? C.paler : C.white, i % 2 ? C.darkLine : C.blue, 2, 20);
    s += `<text x="${it[0] + 130}" y="${it[1] + 48}" class="m" text-anchor="middle">${it[2]}</text>`;
    s += `<text x="${it[0] + 130}" y="${it[1] + 83}" class="s" text-anchor="middle">${it[3]}</text>`;
  });
  s += arrow(500, 455, 585, 455);
  s += `<path d="M765 410 v92 M1055 410 v92 M1345 410 v92" stroke="${C.darkLine}" stroke-width="3" marker-end="url(#arrow-dark)"/>`;
  s += `<text x="1060" y="790" class="b" text-anchor="middle">Manager-based 环境将这些概念拆成可组合管理器；Direct 环境则在类中直接实现</text>`;
  s += end();
  return s;
}

function parallelEvaluation() {
  let s = base("从一次运行到条件分布", "并行的教学价值：同时覆盖初始状态，而不只是把同一个 Demo 跑得更快");
  s += rounded(70, 180, 330, 575, C.white, C.darkLine, 3, 25, true);
  s += `<text x="235" y="230" class="h" text-anchor="middle">单次 Demo</text>`;
  s += rounded(130, 320, 210, 210, C.paler, C.blue, 3, 20);
  s += `<circle cx="235" cy="425" r="35" fill="${C.blue}"/><path d="M190 485 L280 365" stroke="${C.navy}" stroke-width="10"/>`;
  s += `<text x="235" y="590" class="b" text-anchor="middle">只回答“这一次能否成功”</text>`;
  s += arrow(420, 465, 570, 465);
  s += rounded(590, 180, 940, 575, C.paler, C.blue, 3, 25, true);
  s += `<text x="1060" y="230" class="h" text-anchor="middle">条件矩阵与批量统计</text>`;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 5; c++) {
      const x = 650 + c * 165;
      const y = 285 + r * 125;
      const ok = (r + c) % 4 !== 0;
      s += rounded(x, y, 130, 88, C.white, ok ? C.blue : C.darkLine, 2, 14);
      s += `<circle cx="${x + 38}" cy="${y + 44}" r="17" fill="${ok ? C.blue : C.white}" stroke="${ok ? C.blue : C.darkLine}" stroke-width="3"/>`;
      if (ok) s += `<path d="M${x + 29} ${y + 44} l7 8 l13 -18" fill="none" stroke="${C.white}" stroke-width="4"/>`;
      else s += `<path d="M${x + 29} ${y + 35} l18 18 M${x + 47} ${y + 35} l-18 18" stroke="${C.darkLine}" stroke-width="4"/>`;
      s += `<text x="${x + 70}" y="${y + 72}" class="s" text-anchor="middle">env ${r * 5 + c + 1}</text>`;
    }
  }
  s += pill(690, 680, 740, "目标偏移 × 初始姿态 × 摩擦档位 → 成功率与误差分布");
  s += end();
  return s;
}

function g1Timeline() {
  let s = base("G1 摔倒诊断：找最早异常，而不是只看最后一帧", "示意时间序列；阈值应来自具体资产、策略和实验基线");
  const left = 245;
  const right = 1510;
  const top = 190;
  const rowH = 145;
  const labels = ["速度指令 command", "实际速度", "右脚接触", "躯干 roll"];
  labels.forEach((label, i) => {
    const y = top + i * rowH;
    s += `<text x="65" y="${y + 55}" class="m">${label}</text>`;
    s += `<line x1="${left}" y1="${y + 70}" x2="${right}" y2="${y + 70}" stroke="${C.line}" stroke-width="2"/>`;
  });
  s += `<path d="M245 260 L1510 260" fill="none" stroke="${C.blue}" stroke-width="7"/>`;
  s += `<path d="M245 405 C500 405, 630 395, 800 405 S1000 445, 1120 500 S1340 470, 1510 535" fill="none" stroke="${C.navy}" stroke-width="7"/>`;
  s += `<path d="M245 550 L760 550 L780 610 L850 610 L870 550 L1050 550 L1070 620 L1170 620 L1190 550 L1510 550" fill="none" stroke="${C.darkLine}" stroke-width="7"/>`;
  s += `<path d="M245 695 C700 695, 820 690, 990 700 S1120 735, 1210 790 S1380 820, 1510 835" fill="none" stroke="${C.red}" stroke-width="7"/>`;
  s += `<line x1="1055" y1="165" x2="1055" y2="790" stroke="${C.amber}" stroke-width="5" stroke-dasharray="14 10"/>`;
  s += `<line x1="1220" y1="165" x2="1220" y2="845" stroke="${C.red}" stroke-width="5" stroke-dasharray="14 10"/>`;
  s += pill(860, 150, 330, "最早异常：足底接触/速度偏差", "#FFF6E7", C.amber);
  s += pill(1250, 150, 230, "最终症状：摔倒", "#FFF0F0", C.red);
  s += `<text x="875" y="865" class="s" text-anchor="middle">时间 →</text>`;
  s += end();
  return s;
}

function sim2real() {
  let s = base("Sim2Real：仿真契约完整，仍可能把现实写错", "把差距挂回具体契约项，才知道该测什么、改什么");
  const rows = [
    ["模型", "质量 / 惯量 / 几何理想化", "轨迹响应偏差", "系统辨识"],
    ["场景与物理", "接触和摩擦被简化", "抓取滑脱 / 足底打滑", "参数标定与随机化"],
    ["观测", "真值、无噪声、无丢帧", "抖动 / 延迟反应", "噪声与时延注入"],
    ["动作与执行器", "指令即时且线性执行", "超调 / 跟踪滞后", "执行器模型辨识"],
    ["控制链", "采样与通信零时延", "高增益下失稳", "接口对齐与延迟建模"],
    ["视觉域", "纹理、光照、深度理想", "检测/位姿误差", "域随机化与适配"],
  ];
  const xs = [65, 285, 775, 1135];
  const ws = [200, 470, 340, 390];
  ["契约项", "仿真假设", "可测症状", "对应策略"].forEach((h, i) => {
    s += rounded(xs[i], 170, ws[i], 65, i === 0 ? C.navy : C.pale, i === 0 ? C.navy : C.blue, 2, 12);
    s += `<text x="${xs[i] + ws[i] / 2}" y="212" class="m" text-anchor="middle"${i === 0 ? ` style="fill:${C.white}"` : ""}>${h}</text>`;
  });
  rows.forEach((r, ri) => {
    const y = 250 + ri * 92;
    r.forEach((v, ci) => {
      s += rounded(xs[ci], y, ws[ci], 76, ri % 2 ? C.white : C.paler, C.line, 1.5, 10);
      s += `<text x="${xs[ci] + (ci === 0 ? ws[ci] / 2 : 20)}" y="${y + 46}" class="${ci === 0 ? "m" : "b"}" text-anchor="${ci === 0 ? "middle" : "start"}">${v}</text>`;
    });
  });
  s += `<text x="800" y="850" class="b" text-anchor="middle">方法不是越多越好：每一种措施都应对应一个已经观察到或准备验证的错误假设</text>`;
  s += end();
  return s;
}

function experimentLoop() {
  let s = base("把失败变成下一次实验", "证据来自日志、轨迹和重复实验；AI 只负责提出候选假设");
  const items = [
    ["01", "确定基线", "版本 · seed · 初态"],
    ["02", "只改一项", "明确自变量"],
    ["03", "运行前预测", "写下机制假设"],
    ["04", "记录指标", "轨迹 · 误差 · 接触"],
    ["05", "判断证据", "支持 / 否定 / 不充分"],
    ["06", "决定下一步", "复现或新实验"],
  ];
  const y = 325;
  items.forEach((it, i) => {
    const x = 45 + i * 255;
    s += rounded(x, y, 210, 185, i % 2 ? C.white : C.paler, i % 2 ? C.darkLine : C.blue, 3, 22, true);
    s += `<text x="${x + 105}" y="${y + 45}" class="s" text-anchor="middle">${it[0]}</text>`;
    s += `<text x="${x + 105}" y="${y + 88}" class="m" text-anchor="middle">${it[1]}</text>`;
    s += `<text x="${x + 105}" y="${y + 132}" class="s" text-anchor="middle">${it[2]}</text>`;
    if (i < items.length - 1) s += arrow(x + 218, y + 92, x + 247, y + 92);
  });
  s += `<path d="M1435 535 C1435 710, 150 710, 150 535" fill="none" stroke="${C.darkLine}" stroke-width="4" stroke-dasharray="13 10" marker-end="url(#arrow-dark)"/>`;
  s += `<text x="800" y="690" class="b" text-anchor="middle">现象 → 最早异常 → 候选原因 → 单变量验证 → 改进</text>`;
  s += pill(365, 180, 870, "最小复现记录：软件/资产版本 + seed + 两个时钟 + 初始状态 + 指标");
  s += end();
  return s;
}

function platformChoice() {
  let s = base("平台不是排名题：从任务契约反推工具", "同一个项目也可能在不同阶段组合使用多个平台");
  const centerX = 800;
  const centerY = 445;
  s += `<circle cx="${centerX}" cy="${centerY}" r="145" fill="${C.navy}"/>`;
  s += lines(centerX, centerY - 18, ["先问：", "我要观察什么？"], "h", 42, "middle", C.white);
  const cards = [
    [65, 190, 370, 190, "模型、控制、接触", "MuJoCo · PyBullet · Drake", "轻量原型与动力学分析"],
    [1165, 190, 370, 190, "并行学习与评测", "Isaac Lab · Genesis · Brax", "条件分布与批量采样"],
    [65, 585, 370, 190, "ROS 与系统联调", "Gazebo · Webots · CoppeliaSim", "传感器与中间件集成"],
    [1165, 585, 370, 190, "视觉与特定领域", "CARLA · SAPIEN · ManiSkill", "自动驾驶、操作与感知"],
  ];
  cards.forEach((c, i) => {
    s += rounded(c[0], c[1], c[2], c[3], i % 2 ? C.white : C.paler, i % 2 ? C.darkLine : C.blue, 3, 24, true);
    s += `<text x="${c[0] + c[2] / 2}" y="${c[1] + 52}" class="h" text-anchor="middle">${c[4]}</text>`;
    s += `<text x="${c[0] + c[2] / 2}" y="${c[1] + 104}" class="b" text-anchor="middle">${c[5]}</text>`;
    s += `<text x="${c[0] + c[2] / 2}" y="${c[1] + 146}" class="s" text-anchor="middle">${c[6]}</text>`;
    const fromX = c[0] < centerX ? c[0] + c[2] : c[0];
    const fromY = c[1] + c[3] / 2;
    const toX = c[0] < centerX ? centerX - 160 : centerX + 160;
    s += arrow(c[0] < centerX ? toX : fromX, fromY, c[0] < centerX ? fromX : toX, fromY, C.darkLine, true, 3);
  });
  s += pill(470, 760, 660, "其余生态：robosuite · Chrono · Unity/ML-Agents · Unreal · AirSim · Newton");
  s += end();
  return s;
}

async function main() {
  const figures = {
    "05-03-task-contract": taskContract(),
    "05-04-contract-implementation-map": contractMap(),
    "05-06-reach-control-loop": reachLoop(),
    "05-07-pick-place-failures": pickFailures(),
    "05-08-simulation-clocks": clocks(),
    "05-09-isaaclab-environment-flow": isaacFlow(),
    "05-10-parallel-evaluation": parallelEvaluation(),
    "05-12-g1-fall-timeline": g1Timeline(),
    "05-13-sim2real-gaps": sim2real(),
    "05-14-experiment-loop": experimentLoop(),
    "05-15-platform-choice": platformChoice(),
  };
  await Promise.all(Object.entries(figures).map(([name, svg]) => writeFigure(name, svg)));
  process.stdout.write(`Generated ${Object.keys(figures).length} SVG/PNG figure pairs in ${OUT}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack}\n`);
  process.exitCode = 1;
});
