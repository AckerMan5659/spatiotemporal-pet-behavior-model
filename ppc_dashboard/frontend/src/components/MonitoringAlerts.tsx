import React, { useState, useEffect, useRef } from 'react';
import { Video, Wifi, WifiOff, Search, Grid3x3, List, CheckCircle, AlertTriangle, PawPrint, Loader2, Cpu, Zap, Settings, X, Save, Film, History, Calendar, PlayCircle, ArrowLeft, Trash2 } from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';
import { fetchCameraStats, getVideoFeedUrl, BackendStats, getActiveCameras, setActiveCameras, getCameraConfig, updateCameraConfig, getVideoRecords, getRecordVideoUrl, getRecordThumbnailUrl, deleteVideoRecord, VideoRecord } from '../services/api';

interface MonitoringAlertsProps {
    language: Language;
    theme: Theme;
    onOpenLiveMonitoring: (cameraId: string, cameraName: string) => void;
}

interface Camera {
    id: string;
    name: string;
    status: 'online' | 'offline';
    currentBehavior: string;
    confidence: number;
    isAbnormal: boolean;
    fps?: number;
    streamIndex: number;
    activeIds: string[];
    isRecording: boolean;
    imgsz: number;
}

export function MonitoringAlerts({ language, theme, onOpenLiveMonitoring }: MonitoringAlertsProps) {
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [searchQuery, setSearchQuery] = useState('');
    const [highlightedCamera, setHighlightedCamera] = useState<string | null>(null);

    const [cameras, setCameras] = useState<Camera[]>([]);

    const [perfStats, setPerfStats] = useState({ yoloMs: 0, ruleMs: 0, actionMs: 0 });
    const [serverActiveCams, setServerActiveCams] = useState<number[]>([]);

    const [configModalOpen, setConfigModalOpen] = useState<number | null>(null);
    const [configData, setConfigData] = useState({ imgsz: 480, record_labels: [] as string[], record_threshold: 0.70 });
    const [isSaving, setIsSaving] = useState(false);

    const [showHistory, setShowHistory] = useState(false);
    const [records, setRecords] = useState<VideoRecord[]>([]);
    const [isLoadingRecords, setIsLoadingRecords] = useState(false);
    const [playingVideo, setPlayingVideo] = useState<VideoRecord | null>(null);
    const [historyFilters, setHistoryFilters] = useState({ cam_id: 'all', action: 'all', date: '' });

    const BEHAVIOR_OPTIONS = ['Eat', 'Drink', 'Jump', 'Act', 'Rest'];

    useEffect(() => {
        const initCams = async () => {
            const activeList = await getActiveCameras();
            setServerActiveCams(activeList);
        };
        initCams();
    }, []);

    const handleToggleCamera = async (camIndex: number) => {
        const isActive = serverActiveCams.includes(camIndex);
        const newActiveList = isActive ? serverActiveCams.filter(id => id !== camIndex) : [...serverActiveCams, camIndex];
        setServerActiveCams(newActiveList);
        await setActiveCameras(newActiveList);
    };

    const openConfigModal = async (camIndex: number) => {
        setConfigModalOpen(camIndex);
        const currentCfg = await getCameraConfig(camIndex);
        if (currentCfg) {
            setConfigData({
                imgsz: currentCfg.imgsz || 480,
                record_labels: currentCfg.record_labels || [],
                record_threshold: currentCfg.record_threshold || 0.70
            });
        }
    };

    const saveConfig = async () => {
        if (configModalOpen === null) return;
        setIsSaving(true);
        await updateCameraConfig(configModalOpen, configData);
        setIsSaving(false);
        setConfigModalOpen(null);
    };

    const loadRecords = async () => {
        setIsLoadingRecords(true);
        const data = await getVideoRecords(historyFilters);
        setRecords(data);
        setIsLoadingRecords(false);
    };

    useEffect(() => {
        if (showHistory) {
            loadRecords();
        }
    }, [showHistory, historyFilters]);

    const handleDeleteRecord = async (e: React.MouseEvent, id: number) => {
        e.stopPropagation();
        if (window.confirm('確定要永久刪除這筆錄影紀錄與檔案嗎？')) {
            const success = await deleteVideoRecord(id);
            if (success) {
                setRecords(records.filter(r => r.id !== id));
            } else {
                alert('刪除失敗，請檢查後端日誌');
            }
        }
    };

    useEffect(() => {
        if (showHistory) return;

        const loadStats = async () => {
            const data: BackendStats = await fetchCameraStats();
            const newCameras: Camera[] = [];

            const camKeys = Object.keys(data).filter(k => !isNaN(Number(k)));

            // 🌟 核心修復：動態尋找在線攝影機的最高耗時，不再被 "0" 號攝影機卡死
            let maxYolo = 0, maxRule = 0, maxAction = 0;
            camKeys.forEach(k => {
                const s = data[k]?.stats;
                if (s && s.status === 'online') {
                    if (s.yoloMs > maxYolo) maxYolo = s.yoloMs;
                    if (s.ruleMs > maxRule) maxRule = s.ruleMs;
                    if (s.actionMs > maxAction) maxAction = s.actionMs;
                }
            });

            // 只要有任何一個攝影機在運算，就更新性能面板
            setPerfStats({ yoloMs: maxYolo, ruleMs: maxRule, actionMs: maxAction });

            const totalCams = Math.max(camKeys.length, 8);

            for (let i = 0; i < totalCams; i++) {
                const displayId = `CAM-${(i + 1).toString().padStart(2, '0')}`;
                const streamIndex = i;
                const camData = data[streamIndex.toString()];

                let behavior = 'Rest';
                let confidence = 0;
                let fps = 0;
                let isRecording = false;
                let imgsz = 480;

                const isOnline = !!(camData && camData.stats && camData.stats.status !== 'offline');
                let currentActiveIds: string[] = [];

                if (isOnline && camData) {
                    fps = camData.stats.fps || 0;
                    isRecording = camData.stats.isRecording || false;
                    imgsz = camData.stats.imgsz || 480;
                    currentActiveIds = Object.keys(camData.active_states || {});

                    if (currentActiveIds.length > 0) {
                        const mainObj = camData.active_states[currentActiveIds[currentActiveIds.length - 1]];
                        if (mainObj && mainObj.probs) {
                            let maxProb = 0;
                            Object.entries(mainObj.probs).forEach(([key, val]) => {
                                if (BEHAVIOR_OPTIONS.includes(key)) {
                                    if (val > maxProb) { maxProb = val; behavior = key; }
                                }
                            });
                            confidence = Math.round(maxProb * 100);
                        }
                    }
                }

                // 🔥 完全依賴後端提供的動態名稱，若無則降級為 Camera X
                const backendName = camData?.stats?.name;
                const finalCamName = backendName ? backendName : `Camera ${i + 1}`;

                newCameras.push({
                    id: displayId,
                    name: finalCamName,
                    status: isOnline ? 'online' : 'offline',
                    currentBehavior: behavior,
                    confidence,
                    isAbnormal: false,
                    fps,
                    streamIndex,
                    activeIds: currentActiveIds,
                    isRecording,
                    imgsz
                });
            }
            setCameras(newCameras);
        };

        const interval = setInterval(loadStats, 1000);
        return () => clearInterval(interval);
    }, [showHistory]);

    const filteredCameras = cameras.filter((camera) =>
        camera.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        camera.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (showHistory) {
        return (
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button onClick={() => setShowHistory(false)} className={`p-2 rounded-lg transition-colors ${theme === 'dark' ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-white hover:bg-gray-100 text-gray-800'} shadow-sm`}>
                            <ArrowLeft className="w-5 h-5" />
                        </button>
                        <h2 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>歷史檔案庫 (Playback)</h2>
                    </div>
                </div>

                <div className={`p-4 rounded-xl shadow-sm border flex flex-wrap gap-4 items-end ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                    <div className="space-y-1">
                        <label className={`text-xs font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>攝影機</label>
                        <select
                            value={historyFilters.cam_id} onChange={(e) => setHistoryFilters({...historyFilters, cam_id: e.target.value})}
                            className={`block w-40 px-3 py-2 text-sm rounded-lg border ${theme === 'dark' ? 'bg-gray-900 border-gray-700 text-white' : 'bg-gray-50 border-gray-300'}`}
                        >
                            <option value="all">所有攝影機 (All)</option>
                            {[...Array(8)].map((_, i) => <option key={i} value={i+1}>CAM-{i+1}</option>)}
                        </select>
                    </div>
                    <div className="space-y-1">
                        <label className={`text-xs font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>觸發動作</label>
                        <select
                            value={historyFilters.action} onChange={(e) => setHistoryFilters({...historyFilters, action: e.target.value})}
                            className={`block w-40 px-3 py-2 text-sm rounded-lg border ${theme === 'dark' ? 'bg-gray-900 border-gray-700 text-white' : 'bg-gray-50 border-gray-300'}`}
                        >
                            <option value="all">所有動作 (All)</option>
                            {BEHAVIOR_OPTIONS.map(opt => <option key={opt} value={opt}>{t(opt, language)}</option>)}
                        </select>
                    </div>
                    <div className="space-y-1">
                        <label className={`text-xs font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>日期</label>
                        <div className="relative">
                            <input
                                type="date" value={historyFilters.date} onChange={(e) => setHistoryFilters({...historyFilters, date: e.target.value})}
                                className={`block w-40 px-3 py-2 text-sm rounded-lg border pl-9 ${theme === 'dark' ? 'bg-gray-900 border-gray-700 text-white [color-scheme:dark]' : 'bg-gray-50 border-gray-300'}`}
                            />
                            <Calendar className="w-4 h-4 absolute left-3 top-2.5 text-gray-500" />
                        </div>
                    </div>
                </div>

                {isLoadingRecords ? (
                    <div className="h-64 flex flex-col items-center justify-center text-primary">
                        <Loader2 className="w-10 h-10 animate-spin mb-4" />
                        <p>載入紀錄中...</p>
                    </div>
                ) : records.length === 0 ? (
                    <div className="h-64 flex flex-col items-center justify-center border-2 border-dashed border-gray-500/30 rounded-xl">
                        <Film className="w-12 h-12 text-gray-400 mb-2 opacity-50" />
                        <p className={`text-lg font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>找不到符合的錄影紀錄</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                        {records.map((rec) => (
                            <div key={rec.id} className={`relative rounded-xl overflow-hidden border transition-all hover:shadow-lg ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>

                                <button
                                    onClick={(e) => handleDeleteRecord(e, rec.id)}
                                    className="absolute bottom-[4.5rem] right-3 p-2 bg-red-600 hover:bg-red-700 text-white rounded-full shadow-lg z-20 transition-transform hover:scale-110"
                                    title="刪除此紀錄"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>

                                <div className="relative bg-gray-900 aspect-video flex items-center justify-center group cursor-pointer" onClick={() => setPlayingVideo(rec)}>
                                    <img
                                        src={getRecordThumbnailUrl(rec.filename)}
                                        alt="Thumbnail"
                                        className="absolute inset-0 w-full h-full object-cover opacity-80 group-hover:scale-105 transition-transform duration-500"
                                        onError={(e) => { e.currentTarget.style.display = 'none'; }}
                                    />
                                    <Film className="w-12 h-12 text-gray-600 absolute -z-10" />

                                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                        <PlayCircle className="w-16 h-16 text-white opacity-80" />
                                    </div>
                                    <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-sm px-2 py-1 rounded text-xs font-bold text-white shadow-sm">
                                        CAM-{rec.cam_id}
                                    </div>
                                    <div className="absolute top-2 right-2 bg-red-600 px-2 py-1 rounded text-xs font-bold text-white shadow-sm flex items-center gap-1">
                                        <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
                                        {t(rec.trigger_action, language)} ({(rec.max_confidence * 100).toFixed(0)}%)
                                    </div>
                                </div>
                                <div className="p-4 pt-3">
                                    <p className={`text-sm font-bold mb-1 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>{rec.start_time.split(' ')[0]}</p>
                                    <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
                                        時間: {rec.start_time.split(' ')[1]} - {rec.end_time.split(' ')[1]}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {playingVideo && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md p-4">
                        <div className="w-full max-w-4xl bg-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-gray-700">
                            <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
                                <div>
                                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                        <PlayCircle className="w-5 h-5 text-blue-500" />
                                        CAM-{playingVideo.cam_id} 異常事件回放
                                    </h3>
                                    <p className="text-xs text-gray-400 mt-1">觸發動作: {t(playingVideo.trigger_action, language)} | 時間: {playingVideo.start_time}</p>
                                </div>
                                <button onClick={() => setPlayingVideo(null)} className="p-2 bg-gray-700 hover:bg-red-500 rounded-lg text-white transition-colors"><X className="w-5 h-5" /></button>
                            </div>
                            <div className="aspect-video bg-black flex items-center justify-center">
                                <video
                                    src={getRecordVideoUrl(playingVideo.filename)}
                                    controls
                                    autoPlay
                                    className="w-full h-full object-contain"
                                    onError={(e) => console.error("Video Playback Error: ", e)}
                                >
                                    您的瀏覽器不支援 HTML5 影片播放。
                                </video>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-6 relative">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-2">
                <div className="flex flex-row items-center gap-4">
                    <h2 className={`text-xl md:text-2xl font-bold whitespace-nowrap shrink-0 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                        {t('monitoringAlerts', language)}
                    </h2>

                    <div className={`flex flex-wrap items-center gap-2 px-3 py-1.5 rounded-full border shadow-sm backdrop-blur-md shrink-0 mx-auto ${theme === 'dark' ? 'bg-gray-800/80 border-gray-700' : 'bg-white/80 border-gray-200'}`}>
                        <div className="flex items-center gap-1.5 mr-1">
                            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                            <span className={`text-xs font-semibold uppercase tracking-wider whitespace-nowrap ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>系統耗時</span>
                        </div>
                        <div className={`w-px h-3 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`}></div>

                        <div className="flex items-center gap-1">
                            <Cpu className="w-3.5 h-3.5 text-blue-500" />
                            <span className={`text-xs whitespace-nowrap hidden sm:inline ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>YOLO:</span>
                            <span className={`text-xs font-bold font-mono whitespace-nowrap ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{perfStats.yoloMs}ms</span>
                        </div>
                        <div className={`w-px h-3 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`}></div>

                        <div className="flex items-center gap-1">
                            <Settings className="w-3.5 h-3.5 text-orange-500" />
                            <span className={`text-xs whitespace-nowrap hidden sm:inline ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>規則:</span>
                            <span className={`text-xs font-bold font-mono whitespace-nowrap ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{perfStats.ruleMs}ms</span>
                        </div>
                        <div className={`w-px h-3 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`}></div>

                        <div className="flex items-center gap-1">
                            <Zap className="w-3.5 h-3.5 text-purple-500" />
                            <span className={`text-xs whitespace-nowrap hidden sm:inline ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>動作AI:</span>
                            <span className={`text-xs font-bold font-mono whitespace-nowrap ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{perfStats.actionMs}ms</span>
                        </div>
                    </div>
                </div>

                <button
                    onClick={() => setShowHistory(true)}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary hover:bg-primary/90 text-white text-sm font-bold shadow-lg transition-all hover:scale-105 shrink-0"
                >
                    <History className="w-5 h-5" /> 歷史檔案庫
                </button>
            </div>

            <div className={`flex items-center gap-2 overflow-x-auto pb-2 border-b ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'}`}>
                <span className={`text-xs font-bold shrink-0 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>後端推流控制:</span>
                {[0, 1, 2, 3, 4, 5, 6, 7].map((index) => {
                    const isOn = serverActiveCams.includes(index);
                    return (
                        <button
                            key={index}
                            onClick={() => handleToggleCamera(index)}
                            className={`px-3 py-1 text-[10px] font-bold rounded-lg transition-all border whitespace-nowrap shrink-0
                                ${isOn ? 'bg-primary text-white border-primary shadow-md hover:bg-primary/90' : theme === 'dark' ? 'bg-gray-900 text-gray-400 border-gray-700 hover:text-white' : 'bg-gray-100 text-gray-500 border-gray-200 hover:text-gray-800'}
                            `}
                        >
                            CAM-{index + 1} {isOn ? 'ON' : 'OFF'}
                        </button>
                    );
                })}
            </div>

            <div className="flex items-center justify-between mb-4 mt-2">
                <div className="flex items-center gap-2 w-full max-w-sm">
                    <Search className="w-5 h-5 text-gray-500" />
                    <input
                        type="text"
                        placeholder={t('searchCameras', language)}
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className={`w-full px-3 py-1.5 border rounded-lg ${theme === 'dark' ? 'bg-gray-800 border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-200 text-gray-800'}`}
                    />
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={() => setViewMode('grid')} className={`p-1.5 rounded-lg transition-colors ${viewMode === 'grid' ? 'bg-primary/20 text-primary' : 'bg-gray-100 text-gray-500 dark:bg-gray-800'}`}><Grid3x3 className="w-5 h-5" /></button>
                    <button onClick={() => setViewMode('list')} className={`p-1.5 rounded-lg transition-colors ${viewMode === 'list' ? 'bg-primary/20 text-primary' : 'bg-gray-100 text-gray-500 dark:bg-gray-800'}`}><List className="w-5 h-5" /></button>
                </div>
            </div>

            {viewMode === 'grid' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {filteredCameras.map((camera) => (
                        <CameraCard key={camera.id} camera={camera} language={language} theme={theme} highlighted={highlightedCamera === camera.id} onClick={() => onOpenLiveMonitoring(camera.id, camera.name)} onOpenConfig={() => openConfigModal(camera.streamIndex)} />
                    ))}
                </div>
            ) : (
                <div className="space-y-4">
                    {filteredCameras.map((camera) => (
                        <CameraCard key={camera.id} camera={camera} language={language} theme={theme} highlighted={highlightedCamera === camera.id} onClick={() => onOpenLiveMonitoring(camera.id, camera.name)} onOpenConfig={() => openConfigModal(camera.streamIndex)} />
                    ))}
                </div>
            )}

            {configModalOpen !== null && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                    <div className={`w-full max-w-md rounded-2xl shadow-2xl border ${theme === 'dark' ? 'bg-gray-900 border-gray-700 text-gray-100' : 'bg-white border-gray-200 text-gray-800'}`}>
                        <div className="flex items-center justify-between p-4 border-b border-gray-500/20">
                            <h3 className="text-lg font-bold flex items-center gap-2">
                                <Settings className="w-5 h-5 text-primary" />
                                CAM-{configModalOpen + 1} 運算與錄影設定
                            </h3>
                            <button onClick={() => setConfigModalOpen(null)} className="p-1 hover:bg-gray-500/20 rounded-lg transition-colors"><X className="w-5 h-5" /></button>
                        </div>

                        <div className="p-5 space-y-6">
                            <div className="space-y-3">
                                <label className="text-sm font-semibold flex items-center gap-2">
                                    <Cpu className="w-4 h-4 text-blue-500" /> YOLO 推理負載 (雙靜態引擎切換)
                                </label>
                                <div className="grid grid-cols-2 gap-3">
                                    {[480, 640].map((size) => (
                                        <button
                                            key={size}
                                            onClick={() => setConfigData({ ...configData, imgsz: size })}
                                            className={`py-3 text-sm font-bold rounded-xl border transition-all ${
                                                configData.imgsz === size
                                                    ? 'bg-blue-500/10 border-blue-500 text-blue-500 shadow-sm'
                                                    : theme === 'dark'
                                                        ? 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500'
                                                        : 'bg-gray-50 border-gray-200 text-gray-600 hover:border-gray-300'
                                            }`}
                                        >
                                            {size}p
                                            <span className="block text-xs font-normal opacity-70 mt-1">
                                                {size === 480 ? '極速平衡模式 (推薦)' : '高精度高負載模式'}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-3">
                                <label className="text-sm font-semibold flex items-center gap-2">
                                    <Film className="w-4 h-4 text-red-500" /> 觸發錄影動作 (Auto-Record)
                                </label>
                                <div className="flex flex-wrap gap-2">
                                    {BEHAVIOR_OPTIONS.map(label => {
                                        const isSelected = configData.record_labels.includes(label);
                                        return (
                                            <button
                                                key={label}
                                                onClick={() => {
                                                    const newLabels = isSelected ? configData.record_labels.filter(l => l !== label) : [...configData.record_labels, label];
                                                    setConfigData({ ...configData, record_labels: newLabels });
                                                }}
                                                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${isSelected ? 'bg-red-500 text-white border-red-500 shadow-md' : theme === 'dark' ? 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500' : 'bg-gray-50 border-gray-200 text-gray-600 hover:border-gray-300'}`}
                                            >
                                                {t(label, language)}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            <div className="space-y-3">
                                <div className="flex justify-between items-end">
                                    <label className="text-sm font-semibold">觸發靈敏度閥值 (Threshold)</label>
                                    <span className="text-xs font-bold text-primary">{Math.round(configData.record_threshold * 100)}%</span>
                                </div>
                                <input
                                    type="range"
                                    min="0.4" max="0.95" step="0.05"
                                    value={configData.record_threshold}
                                    onChange={(e) => setConfigData({ ...configData, record_threshold: parseFloat(e.target.value) })}
                                    className="w-full accent-primary"
                                />
                                <div className="flex justify-between text-[10px] text-gray-500">
                                    <span>易誤報 (40%)</span><span>嚴格 (95%)</span>
                                </div>
                            </div>
                        </div>

                        <div className="p-4 border-t border-gray-500/20 flex justify-end gap-3">
                            <button onClick={() => setConfigModalOpen(null)} className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-500/10 hover:bg-gray-500/20 transition-colors">取消</button>
                            <button onClick={saveConfig} disabled={isSaving} className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary/90 flex items-center gap-2 transition-colors disabled:opacity-50">
                                {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />} 套用設定
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

function CameraCard({ camera, language, theme, highlighted, onClick, onOpenConfig }: any) {
    const [videoError, setVideoError] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [retryKey, setRetryKey] = useState(0);
    const imgRef = useRef<HTMLImageElement>(null);
    const videoFeedUrl = getVideoFeedUrl(camera.streamIndex);

    useEffect(() => {
        const currentImg = imgRef.current;
        return () => { if (currentImg) { currentImg.src = ''; } };
    }, []);

    useEffect(() => {
        if (camera.status === 'online') {
            setVideoError(false);
            const timer = setTimeout(() => { setIsLoading(false); }, 1500);
            return () => clearTimeout(timer);
        } else {
            setIsLoading(true);
        }
    }, [camera.status, camera.streamIndex]);

    useEffect(() => {
        if (camera.status === 'online' && videoError) {
            setVideoError(false);
            setIsLoading(true);
            setRetryKey(prev => prev + 1);
        }
    }, [camera.status, videoError]);

    const apiOffline = camera.status === 'offline';
    const isActuallyOffline = apiOffline || videoError;
    const isConnecting = !apiOffline && !videoError && isLoading;
    const isLive = !apiOffline && !videoError && !isLoading;

    return (
        <div
            id={`camera-${camera.id}`}
            onClick={onClick}
            className={`rounded-xl shadow-sm overflow-hidden transition-all cursor-pointer hover:shadow-lg hover:scale-[1.02]
              ${camera.isRecording ? 'border-2 border-red-500 shadow-md shadow-red-500/20' : camera.isAbnormal ? 'border-4 border-red-500 shadow-lg' : theme === 'dark' ? 'border border-gray-700 bg-gray-800' : 'border border-gray-200 bg-white'}
              ${highlighted ? 'ring-4 ring-primary shadow-2xl scale-105' : ''}
            `}
        >
            <div className="relative bg-black aspect-video group">
                {!apiOffline && !videoError && (
                    <img
                        ref={imgRef}
                        key={retryKey}
                        src={`${videoFeedUrl}?k=${retryKey}`}
                        alt={camera.name}
                        className={`w-full h-full object-cover transition-transform duration-700 group-hover:scale-105 ${isLoading ? 'hidden' : 'block'}`}
                        onError={() => { setIsLoading(false); setVideoError(true); }}
                    />
                )}
                {isActuallyOffline && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900 z-10">
                        <WifiOff className="w-12 h-12 text-gray-600 mb-2" />
                        <p className="text-sm text-gray-500">{t('offline', language)}</p>
                    </div>
                )}
                {isConnecting && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800 z-20">
                        <Loader2 className="w-8 h-8 text-primary animate-spin mb-2" />
                        <p className="text-xs text-gray-400">連線引擎中...</p>
                    </div>
                )}
                {isLive && (
                    <>
                        <div className="absolute top-3 left-3 flex items-center gap-2 z-10">
                            {camera.isRecording ? (
                                <div className="flex items-center gap-1.5 bg-red-600/90 backdrop-blur-md px-3 py-1 rounded-full shadow-lg border border-red-400/50">
                                    <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                    <span className="text-xs font-bold text-white tracking-widest">REC</span>
                                </div>
                            ) : (
                                <div className="flex items-center gap-1.5 bg-green-500/80 backdrop-blur-md px-3 py-1 rounded-full shadow-md">
                                    <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                                    <span className="text-xs font-bold text-white tracking-wider">LIVE</span>
                                </div>
                            )}
                        </div>

                        <div className="absolute top-3 right-3 flex items-center gap-2 z-10">
                            <div className="bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-[10px] text-gray-300 border border-gray-600/50 font-mono">
                                {camera.imgsz}p
                            </div>
                            <div className="bg-black/70 backdrop-blur-sm px-3 py-1 rounded-lg text-xs font-medium text-white shadow-md">
                                {camera.id}
                            </div>
                        </div>

                        {camera.activeIds.length > 0 && (
                            <div className="absolute bottom-3 left-3 bg-black/50 backdrop-blur-sm px-2 py-1 rounded text-xs text-white z-10 flex items-center gap-1">
                                <PawPrint className="w-3 h-3" />
                                {camera.activeIds.length}
                            </div>
                        )}
                        <div className="absolute bottom-3 right-3 bg-black/50 backdrop-blur-sm px-2 py-1 rounded text-xs text-white z-10">
                            {camera.fps} FPS
                        </div>
                    </>
                )}
            </div>

            <div className="p-4">
                <div className="flex items-center justify-between mb-3">
                    <h3 className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{camera.name}</h3>
                    <div className="flex gap-2">
                        <button
                            onClick={(e) => { e.stopPropagation(); onOpenConfig(); }}
                            className={`p-1.5 rounded-lg transition-colors border ${theme === 'dark' ? 'bg-gray-700/50 border-gray-600 hover:bg-gray-600 text-gray-300' : 'bg-gray-100 border-gray-200 hover:bg-gray-200 text-gray-600'}`}
                            title="運算與錄影設定"
                        >
                            <Settings className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                {!isActuallyOffline ? (
                    <>
                        <div className="mb-3">
                            <div className="flex items-center justify-between mb-1">
                                <span className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{t('behavior', language)}:</span>
                                <span className={`text-sm font-medium px-2 py-0.5 rounded shadow-sm ${camera.isRecording ? 'bg-red-500 text-white animate-pulse' : 'bg-primary text-white'}`}>
                                    {t(camera.currentBehavior, language)}
                                </span>
                            </div>
                        </div>
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <span className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{t('confidence', language)}:</span>
                                <span className={`text-sm font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{camera.confidence}%</span>
                            </div>
                            <div className={`w-full rounded-full h-2 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}><div className={`h-2 rounded-full ${camera.isRecording ? 'bg-red-500' : 'bg-primary'} transition-all`} style={{ width: `${camera.confidence}%` }} /></div>
                        </div>
                    </>
                ) : (
                    <div className="h-14 flex items-center justify-center border-t border-dashed border-gray-700 mt-2"><span className="text-xs text-gray-500 italic">No Signal</span></div>
                )}
            </div>
        </div>
    );
}