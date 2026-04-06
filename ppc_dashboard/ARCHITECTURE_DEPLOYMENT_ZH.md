# ppc_dashboard系统架构与部署说明_2026_3_19

本文基于当前仓库代码整理，目标是帮助开发、部署、联调和后续维护时快速理解整个系统的结构、运行方式和已存在的边界。

## 1. 系统整体概览

这是一个“前后端分离 + 本地视频推理 + 本地录像归档”的宠物监控系统，当前仓库不是 monorepo，也没有统一的根级应用启动器，而是由以下 3 个部分协同组成：

- `frontend/`：前端界面，基于 Vite + React 18 + TypeScript，负责监控大盘、告警页、实时监控页、数据分析页、设置页等。
- `backend/`：后端推理服务，基于 Flask + Waitress，负责摄像头接入、目标检测、行为识别、录像、数据库记录、接口输出、MJPEG 视频流。
- `go2rtc/`：本地 RTSP 中继组件，用于把某些摄像头源转换为本地 RTSP 地址，供后端默认参数直接拉流。

系统的典型运行方式是：

1. 摄像头视频先进入 `go2rtc` 或直接被后端读取。
2. 后端做检测、规则分析、动作识别、状态融合。
3. 后端把每路摄像头的 `stats / logs / active_states` 暴露成 HTTP API，并提供 `/video_feed/<cam_id>` MJPEG 画面流。
4. 前端通过轮询 `/stats` 和调用配置/录像相关 API，驱动整个 UI。

## 2. 目录职责

### 根目录

- `ARCHITECTURE_DEPLOYMENT_ZH.md`：本文档。
- `requirements.txt`：根级 Python 依赖清单。
- `ppc_dasboard_env.bat`：Windows 环境一键安装脚本，会申请管理员权限并自动安装 Python / Node / pip 依赖，不是日常启动命令。

### 前端目录 `frontend/`

- `package.json`：前端依赖与脚本。
- `vite.config.ts`：Vite 配置，端口为 `3000`，构建输出到 `build/`。
- `src/App.tsx`：应用主入口，控制页面切换。
- `src/services/api.ts`：前后端接口契约与请求封装。
- `src/components/`：主要页面和可视化组件。
- `src/utils/translations.ts`：多语言文案和行为标签映射。

### 后端目录 `backend/`

- `yolo_infer_xiaomi_win_deploy.py`：后端总控入口，整个系统最核心的单文件。
- `config.yaml`：动作识别模式、类别、模型路径、检测器参数等核心配置来源。
- `OpenVINO_convert.py`：YOLO OpenVINO 导出/量化工具，不是日常运行主程序。
- `core/`：规则引擎、融合代理、碗识别等核心逻辑。
- `modules/`：动作识别模型适配与网络结构。
- `model/`：模型文件目录。
- `records.db`：SQLite 录像索引数据库。
- `records/`：录像文件和缩略图输出目录。

### 中继目录 `go2rtc/`

- `go2rtc.exe`：Windows 下的 RTSP 中继程序。
- `go2rtc.yaml`：本机中继配置，包含本地环境相关信息，应视为敏感配置。

## 3. 系统运行拓扑

```text
[IPC / USB Camera]
        |
        |  直接拉流 or 经 go2rtc 中继
        v
[go2rtc: RTSP 8554 / API 1984]
        |
        v
[backend/yolo_infer_xiaomi_win_deploy.py]
    |- 硬件探测与策略分派
    |- 摄像头拉流 / 解码
    |- YOLO 检测与跟踪
    |- BowlManager 碗类型识别
    |- RuleEngine 规则层候选状态
    |- ActionRecognizer 动作模型识别
    |- FusionAgent 状态融合 + FSM 平滑
    |- SQLite 录像索引写入
    |- Flask API / MJPEG 输出
        |
        | HTTP 9527
        +--> /stats
        +--> /api/active_cams
        +--> /api/config/<cam_id>
        +--> /api/records
        +--> /video_feed/<cam_id>
        +--> /records/<path>
        |
        v
[frontend React SPA :3000]
    |- Dashboard
    |- MonitoringAlerts
    |- LiveMonitoring
    |- DataVisualization
    |- PetDataManagement
    |- Settings
```

## 4. 前端架构说明

## 4.1 应用壳层

前端入口非常直接：

- `frontend/src/main.tsx` 只负责挂载 `App`。
- `frontend/src/App.tsx` 通过 React 本地状态切换不同页面。
- 当前没有使用 React Router，因此页面切换不是 URL 路由，而是应用内部状态控制。

这意味着：

- 页面跳转逻辑集中在 `App.tsx`。
- 若后续要做真正的路由、浏览器回退、权限路由，需要重构当前结构。

## 4.2 前端主要页面

### 1) Dashboard

文件：`frontend/src/components/Dashboard.tsx`

职责：

- 展示在线摄像头总览。
- 轮询后端 `/stats`。
- 统计各摄像头当前主行为、置信度、活跃目标数。
- 基于最近轮询结果构造趋势图和分布图。
- 支持跳转到实时监控页和监控告警页。

特点：

- 强依赖 `/stats` 轮询。
- 会把后端 `active_states` 转成前端图表数据。
- 对性能时间采用“在线摄像头中的最大耗时”展示，而不是固定看某一路。

### 2) MonitoringAlerts

文件：`frontend/src/components/MonitoringAlerts.tsx`

职责：

- 展示多路摄像头监控卡片。
- 控制启停某些摄像头。
- 调用后端 `/api/active_cams` 管理当前激活摄像头列表。
- 调用 `/api/config/<cam_id>` 修改单路摄像头配置（如 `imgsz`、录像触发标签、触发阈值）。
- 调用 `/api/records` 查看录像记录。
- 调用 `/api/records/<record_id>` 删除录像。

这是当前前端中与后端运行态联动最深的页面之一。

### 3) LiveMonitoring

文件：`frontend/src/components/LiveMonitoring.tsx`

职责：

- 展示单路摄像头实时画面。
- 使用 `/video_feed/<cam_id>` 获取 MJPEG 画面流。
- 每 500ms 轮询 `/stats`，同步行为、置信度、FPS、在线状态。
- 支持从界面直接开关该摄像头是否在后端激活列表中。

重要实现细节：

- 这里不是 WebRTC，也不是 HLS，而是通过 `<img src="/video_feed/...">` 的 MJPEG 方式展示视频。
- `frontend/src/services/api.ts` 中 `getVideoFeedUrl()` 在本地开发模式下会把请求分散到 `127.0.0.1 ~ 127.0.0.8` 的 loopback 地址。这是前端当前的一个特殊实现，改动域名、代理、反向代理时必须一起考虑。
- `handleSnapshot` / `handleToggleRecording` 等若干方法目前是占位实现，并未真正接通后端完整能力。

### 4) DataVisualization

文件：`frontend/src/components/DataVisualization.tsx`

职责：

- 从 `/stats` 聚合实时统计数据。
- 生成行为分布、雷达图、摄像头性能图等图表。
- 用 `localStorage` 保存监控开始时间，计算监控时长。

边界说明：

- 该页面部分数据来自真实轮询，例如 `/stats` 中的行为和置信度。
- 也有部分内容是前端本地衍生值或静态样例值，例如 `monthlyData`。
- 因此它更接近“实时大屏 + 演示化分析”，不是完整的报表系统。

### 5) PetDataManagement

文件：`frontend/src/components/PetDataManagement.tsx`

职责：

- 展示宠物资料、详情、照片和视频等界面。

边界说明：

- 当前主要是本地 `useState` 和 mock 数据逻辑。
- 没有看到对应的后端宠物资料 API。
- 因此这个模块目前更偏前端静态管理界面，不是真正落库或后端联动模块。

### 6) Settings

文件：`frontend/src/components/Settings.tsx`

职责：

- 提供个人资料、安全设置、通知设置、系统偏好等界面。

边界说明：

- 当前保存动作是前端本地状态反馈，不会持久化到后端。
- 暂未接入专门的后端设置接口。

## 4.3 前端与后端的契约层

文件：`frontend/src/services/api.ts`

这里是最关键的“前后端契约层”。

当前固定的后端基地址：

```ts
export const API_BASE_URL = 'http://127.0.0.1:9527';
```

该文件封装了：

- `fetchCameraStats()` -> `/stats`
- `getActiveCameras()` / `setActiveCameras()` -> `/api/active_cams`
- `getCameraConfig()` / `updateCameraConfig()` -> `/api/config/<cam_id>`
- `getVideoRecords()` / `deleteVideoRecord()` -> `/api/records`
- `getRecordVideoUrl()` / `getRecordThumbnailUrl()` -> `/records/<path>`
- `getVideoFeedUrl()` -> `/video_feed/<cam_id>`

如果要调整部署域名、接口前缀、反向代理路径、鉴权方案，优先从这个文件入手。

## 4.4 前端行为标签同步点

文件：`frontend/src/utils/translations.ts`

前端里对行为标签有硬编码映射，例如：

- `Eat`
- `Drink`
- `Rest`
- `Jump`
- `Act`

这与后端输出的 `probs` 键名耦合很深。

因此只要后端行为类别变动，至少要同步检查：

- `backend/config.yaml`
- `backend/modules/action_recognizer.py`
- `frontend/src/utils/translations.ts`
- `Dashboard.tsx`
- `MonitoringAlerts.tsx`
- `LiveMonitoring.tsx`
- `DataVisualization.tsx`

## 5. 后端架构说明

## 5.1 后端总入口

文件：`backend/yolo_infer_xiaomi_win_deploy.py`

这个文件承担了几乎整个后端系统的主流程，包括：

- 运行前环境检查
- 硬件探测
- 策略选择
- 多进程生命周期管理
- Flask API 创建
- MJPEG 推流
- SQLite 录像索引写入
- 录像清理线程
- 系统负载调速器

换句话说，这不是“薄控制器 + 多模块服务”的后端结构，而是一个“总控脚本型后端”。

## 5.2 启动前检查

后端启动时会先执行：

- 核心依赖存在性检查
- 在低配 CPU 场景下尝试进行一定程度的 OpenVINO 自愈处理

这部分逻辑说明当前系统非常偏“单机部署 + 面向特定 Windows 环境做兼容”设计。

## 5.3 硬件分层与策略工厂

后端会根据 CPU 核心数、是否有 CUDA、是否是 ARM 架构，把机器分到不同层级：

- `TIER_CUDA`
- `TIER_JETSON`
- `TIER_HIGH`
- `TIER_LOW`

然后 `StrategyFactory` 会给出一组运行策略，包括：

- `target_fps`
- `imgsz`
- `device`
- `model_format`
- `max_cams_suggested`
- `max_targets_per_cam`
- `processing_mode`

这一步很关键，因为它直接决定：

- 用 CPU 还是 CUDA
- 用 OpenVINO 还是 PyTorch / TensorRT
- 每路最多追踪多少只目标
- 整体采用并行多进程还是串行集中引擎

## 5.4 两种核心执行模式

### 模式 A：并行模式

入口函数：`worker_process(...)`

适用：

- CUDA 机器
- 较高性能 CPU 机器

特点：

- 每个激活摄像头一个独立 worker 进程。
- 主进程通过共享状态决定启停哪些 worker。
- 更适合多路摄像头并发。

### 模式 B：串行集中模式

入口函数：`centralized_engine_process(...)`

适用：

- 低配机器
- 资源紧张环境

特点：

- 只有一个集中推理引擎进程。
- 轮流处理多个摄像头。
- 资源更稳定，但并发能力较弱。

## 5.5 摄像头动态启停机制

后端不是固定启动所有摄像头，而是有一个“期望激活列表”：

- 存在共享变量 `desired_cams`
- 通过 `/api/active_cams` 暴露给前端
- 前端可以动态开启/关闭某路
- 在并行模式下，主进程会按需拉起或关闭对应 worker

这使系统具备“按需激活摄像头、降低资源占用”的能力。

## 5.6 每路视频的推理流水线

单路摄像头的大体处理顺序如下：

1. 拉流 / 解码视频帧
2. 运行 YOLO 检测与跟踪
3. 识别碗的位置与碗类型
4. 规则引擎判断当前运动和交互候选状态
5. 动作识别模型输出行为概率
6. 融合规则层 + 模型层结果
7. 经 FSM 进行状态平滑
8. 更新共享状态、渲染可视化框、必要时触发录像
9. 输出 MJPEG 帧给前端

其中各模块职责如下。

### 1) `BowlManager`

文件：`backend/core/bowl_manager.py`

职责：

- 从 YOLO 检测结果里找碗（class 0）。
- 根据碗区域颜色粗略区分是 `Food` 还是 `Water`。

它不是基于训练模型，而是基于颜色阈值的规则方法。

### 2) `RuleEngine`

文件：`backend/core/rule_engine.py`

职责：

- 根据目标速度、方向、头部框与碗的重叠程度，推导候选状态。
- 输出类似：
  - `RESTING`
  - `ACTIVE`
  - `INTERACT_CANDIDATE`

它相当于“物理/几何层”的先验规则判断。

### 3) `ActionRecognizer`

文件：`backend/modules/action_recognizer.py`

职责：

- 读取 `backend/config.yaml`
- 根据 `mode` 选择当前任务配置（如 `normal` / `abnormal`）
- 加载 ONNX 或 PyTorch 模型
- 对目标裁剪区域进行预处理并输出行为概率

该模块支持：

- ONNX Runtime 推理
- PyTorch 推理

也就是说，动作识别模型的真正来源和类别定义都以 `config.yaml` 为准，而不是前端常量。

### 4) `FusionAgent`

文件：`backend/core/fusion_agent.py`

职责：

- 把规则引擎的结果和动作识别模型结果进行融合。
- 用 FSM 做状态平滑，避免行为状态频繁抖动。
- 保存最近一次概率、状态、状态切换信息。

这是“最终行为状态”的核心生成模块。

## 5.7 全局调速器

函数：`system_governor_thread(...)`

职责：

- 独立监控系统 CPU 负载。
- 根据硬件层级动态调节 `global_penalty`。
- 间接影响目标 FPS 和推理节奏。

它的意义是：

- 机器负载高时自动降速
- 机器负载低时尽量恢复性能
- 对老机器和高配机器采用不同的调速阈值

因此该系统并不是完全固定帧率运行，而是有运行时自适应节流策略。

## 5.8 MJPEG 推流与帧分发

相关实现：`StreamBroker`

职责：

- 每路摄像头维护一个帧队列。
- `StreamBroker` 在后台线程中不断读取最新 JPEG 帧。
- Flask 的 `/video_feed/<cam_id>` 通过 `multipart/x-mixed-replace` 返回 MJPEG 流。

这套机制的特点：

- 简单、浏览器兼容性好
- 易于本地部署
- 但并不是低带宽、高并发的最佳生产级视频方案

## 5.9 录像与数据库

### 录像触发

当某路摄像头满足以下条件时会触发录像：

- 当前行为属于该摄像头配置里的 `record_labels`
- 且对应概率超过 `record_threshold`

### 录像输出

后端会生成：

- 视频文件：`records/<日期>/videos/*.webm`
- 缩略图：`records/<日期>/thumbnails/*.jpg`

### 索引数据库

使用 SQLite：`backend/records.db`

表：`video_records`

记录字段包括：

- 摄像头 ID
- 文件名
- 开始时间 / 结束时间
- 触发动作
- 最大置信度
- 状态

### 自动清理

后端有清理线程，会按保留天数删除旧录像和旧数据库记录。

当前代码中的保留天数为：

- `KEEP_RECORDS_DAYS = 7`

## 6. 核心配置说明

## 6.1 `backend/config.yaml`

这是整个行为识别链路的核心配置文件。

当前可配置内容包括：

- 模式：`normal` / `abnormal`
- 行为类别名称 `class_names`
- 行为模型路径 `output_model_name`
- 识别器参数 `recognizer`
- YOLO 检测器参数 `detector`
- 推理管线参数 `pipeline`

如果出现以下改动，优先检查它：

- 类别数量变了
- 模型路径变了
- 正常/异常模式切换
- detector 输入尺寸调整
- 跟踪器调整

## 6.2 Python 依赖清单存在两份

仓库当前有两份 Python 依赖：

- 根目录 `requirements.txt`
- `backend/requirements.txt`

它们版本并不一致。

这意味着：

- 环境安装时必须先明确“以哪份为准”。
- 不能默认两份完全等价。
- 修改依赖时最好同步梳理，不然很容易出现“某台机器可跑、另一台机器不行”的情况。

## 6.3 敏感与运行时文件

以下内容通常不应随意修改或提交：

- `go2rtc/go2rtc.yaml`：本机摄像头中继配置，通常包含局域网设备信息或本地密钥类内容。
- `backend/model/`：模型二进制与推理资产。
- `backend/records.db`：运行态数据库。
- `backend/records/`：录像与缩略图产物。
- `frontend/node_modules/`、`.venv/`：依赖产物。

## 7. 部署方式说明

## 7.1 当前仓库更适合的部署模式

从当前代码和目录结构来看，这个项目目前更适合：

- 单机部署
- Windows 本地部署
- 前后端手动分别启动
- 可选搭配本地 `go2rtc` 中继

当前仓库中没有发现以下标准化部署资产：

- `Dockerfile`
- `docker-compose.yml`
- systemd service 文件
- Nginx 反向代理配置
- CI/CD 部署脚本

所以它目前不是“开箱即用的容器化生产部署项目”，而是“适合本地机房 / 边缘设备 / Windows 工控机式部署”的工程。

## 7.2 端口与服务关系

默认端口如下：

- 前端 Vite 开发服务：`3000`
- 后端 Flask/Waitress：`9527`
- go2rtc RTSP：`8554`
- go2rtc API：`1984`

典型访问关系：

- 浏览器访问前端：`http://127.0.0.1:3000`
- 前端请求后端：`http://127.0.0.1:9527`
- 后端拉本地 RTSP：`rtsp://127.0.0.1:8554/...`

## 7.3 推荐启动顺序

推荐按下面顺序启动：

### 第一步：准备 Python 环境

优先进入 `backend/`，按你的依赖策略安装：

```bash
python -m pip install -r requirements.txt
```

如果你的环境是按根目录脚本安装的，也可能使用根级 `requirements.txt`。这一点需要结合你的机器环境确认。

### 第二步：准备前端依赖

在 `frontend/` 下：

```bash
npm install
```

### 第三步：检查后端核心配置

重点检查：

- `backend/config.yaml`
- 模型路径是否真实存在
- `mode` 是否符合当前场景
- detector 模型路径是否有效

### 第四步：如使用 RTSP 中继，先启动 `go2rtc`

当前仓库包含 `go2rtc/go2rtc.exe` 与 `go2rtc/go2rtc.yaml`，说明默认部署形态依赖本地中继。

如果你的后端采用默认 RTSP 参数，那么必须先保证本地 `8554` 上对应流已经存在。

注意：

- `go2rtc/go2rtc.yaml` 是环境相关配置。
- 不要把其中的敏感信息写入文档或提交到公共仓库。

### 第五步：启动后端

在 `backend/` 下：

```bash
python yolo_infer_xiaomi_win_deploy.py
```

启动后，后端会：

- 初始化数据库
- 分析硬件
- 选择运行策略
- 启动 Flask/Waitress
- 初始化共享状态
- 默认激活一部分摄像头列表

### 第六步：启动前端

在 `frontend/` 下：

```bash
npm run dev
```

### 第七步：打开浏览器验证

前往前端开发服务页面后，优先检查：

- Dashboard 是否能显示摄像头卡片
- 监控页是否能开关摄像头
- 实时监控页是否能看到画面与行为状态
- 历史录像页是否能拉到记录

## 7.4 常见后端启动方式

### 方式 A：使用 RTSP 中继源

后端入口支持：

- `--source rtsp`
- `--rtsp_url_1 ... --rtsp_url_8 ...`
- `--num_cams`

适合：

- IPC 摄像头
- go2rtc 转出的本地 RTSP 流

### 方式 B：使用本地摄像头索引

后端入口支持：

- `--source local`
- `--cam_index_1 ... --cam_index_8 ...`

适合：

- USB 摄像头
- 本地直接接入的视频设备

### 方式 C：强制指定设备与帧率

后端入口支持：

- `--device auto|cuda|cpu`
- `--hwaccel auto|qsv|none`
- `--max_fps`

适合：

- 调试性能问题
- 某些机器自动识别策略不理想时做人工干预

## 7.5 Waitress 与 Flask

后端启动时：

- 如果安装了 `waitress`，优先使用 Waitress 监听 `0.0.0.0:9527`
- 如果没有 `waitress`，才回退为 Flask 自带服务器

因此部署环境里最好保证 `waitress` 可用。

## 8. API 接口说明

## 8.1 `/stats`

作用：

- 返回所有摄像头的共享运行态数据。

每路核心结构包括：

- `stats`：FPS、耗时、状态、名称、是否录像等
- `logs`：本轮识别日志
- `active_states`：当前活跃目标及其状态概率

这是前端最重要的数据源。

## 8.2 `/api/active_cams`

作用：

- 获取当前激活摄像头列表
- 设置当前激活摄像头列表

前端监控页和实时监控页都会用它来控制后端是否真正处理某一路。

## 8.3 `/api/config/<cam_id>`

作用：

- 获取单路摄像头运行配置
- 更新单路摄像头运行配置

当前支持的关键字段包括：

- `imgsz`
- `record_labels`
- `record_threshold`

## 8.4 `/api/records`

作用：

- 查询历史录像记录
- 支持按摄像头、动作、日期过滤

## 8.5 `/api/records/<record_id>`

作用：

- 删除某条录像记录
- 同时删除对应视频文件和缩略图

## 8.6 `/records/<path>`

作用：

- 提供录像文件和缩略图的静态访问

## 8.7 `/video_feed/<cam_id>`

作用：

- 返回 MJPEG 实时视频流

这是实时监控页面核心依赖的接口。

## 9. 实际使用建议

## 9.1 本地开发联调建议

如果你在做界面改动：

- 重点看 `frontend/src/App.tsx`
- 接口改动必须同步 `frontend/src/services/api.ts`
- 行为标签改动必须同步 `frontend/src/utils/translations.ts`

如果你在做后端推理改动：

- 优先看 `backend/yolo_infer_xiaomi_win_deploy.py`
- 再看 `backend/core/` 和 `backend/modules/action_recognizer.py`
- 行为类别和模型路径变动一定先确认 `backend/config.yaml`

## 9.2 性能调优建议

如果机器跑不动，先考虑：

- 降低 `--max_fps`
- 强制 `--device cpu` 或 `--device cuda`
- 调低单路 `imgsz`
- 减少激活摄像头数量
- 检查是否误用了不匹配的模型格式

因为当前系统已经内置硬件分层和动态调速，所以性能问题通常不是单一参数导致，而是“机器能力 + 模型格式 + 摄像头数量 + 前端轮询频率”共同决定。

## 9.3 行为识别联调建议

如果前端看得到画面，但行为标签不对：

优先检查：

1. `backend/config.yaml` 里的 `class_names`
2. `backend/modules/action_recognizer.py` 是否加载了正确模型
3. `backend/core/fusion_agent.py` 的融合阈值与 FSM 平滑
4. 前端 `translations.ts` 和各页面里的硬编码行为列表

## 9.4 录像功能联调建议

如果监控画面正常，但没有录像：

优先检查：

1. 某路摄像头的 `record_labels` 是否已配置
2. `record_threshold` 是否过高
3. `records/` 目录是否可写
4. `records.db` 是否可访问
5. 当前行为概率是否真正超过阈值

## 10. 当前系统边界与已知限制

从现有代码看，当前系统具备较完整的“本地实时监控 + 推理 + 录像 + 可视化”骨架，但仍有以下边界：

- 没有标准化测试体系；未发现 pytest、Vitest、Jest 等测试配置。
- 没有标准化 lint 流程；前端 `package.json` 未定义 lint script。
- 没有容器化部署资产。
- `PetDataManagement` 主要是前端 mock 逻辑，并未接入真实后端数据管理。
- `Settings` 主要是前端本地状态保存，不是后端持久化设置系统。
- `LiveMonitoring` 中部分媒体功能仍是占位实现。
- 前端若干分析数据是本地衍生值，不是完整报表后端。
- Python 依赖清单存在两份且版本不一致，后续最好统一。

## 11. 适合后续补强的方向

如果后续要把这个项目进一步工程化，比较自然的方向包括：

- 统一 Python 依赖清单
- 增加前后端统一启动说明或启动脚本
- 为前端补充环境变量化的 API 地址配置
- 为后端拆分总控脚本，降低单文件复杂度
- 给真实业务模块补充测试用例
- 给 `PetDataManagement` / `Settings` 增加真实后端接口
- 补充生产部署方案（容器化、进程守护、日志归档、代理层等）

## 12. 快速结论

如果只用一句话概括当前系统：

这是一个以 `backend/yolo_infer_xiaomi_win_deploy.py` 为核心总控、以 `/stats + /video_feed` 为数据主干、以 React 轮询界面为展示层、面向 Windows 单机本地部署场景设计的多摄像头宠物行为监控系统。
