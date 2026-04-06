import React, { useState, useEffect, useRef } from 'react';
import {
    ArrowLeft, Camera, Circle, Activity, Wifi, Settings,
    Square, AlertTriangle, FileImage, FileVideo, Download, Trash2, Play,
    Power, WifiOff, Loader2
} from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';
import { getVideoFeedUrl, fetchCameraStats, getActiveCameras, setActiveCameras } from '../services/api';

interface LiveMonitoringProps {
    language: Language;
    theme: Theme;
    cameraId: string;
    cameraName: string;
    onBack: () => void;
}

interface MediaItem {
    id: string;
    type: 'image' | 'video';
    url: string;
    timestamp: number;
}

interface AlertEvent {
    id: number;
    time: string;
    message: string;
    level: 'high' | 'medium';
}

export function LiveMonitoring({ language, theme, cameraId, cameraName, onBack }: LiveMonitoringProps) {
    const streamIndex = parseInt(cameraId.split('-')[1]) - 1 || 0;
    const videoUrl = getVideoFeedUrl(streamIndex);

    const [currentStats, setCurrentStats] = useState({ behavior: 'Rest', confidence: 0 });
    const [fps, setFps] = useState(0);
    const [showSettings, setShowSettings] = useState(false);
    const [isRecording, setIsRecording] = useState(false);

    const [apiOffline, setApiOffline] = useState(true);
    const [isLoading, setIsLoading] = useState(true);
    const [videoError, setVideoError] = useState(false);
    const [retryKey, setRetryKey] = useState(0);

    const [serverActiveCams, setServerActiveCams] = useState<number[]>([]);
    const [isActiveOnServer, setIsActiveOnServer] = useState(false);

    const [mediaGallery, setMediaGallery] = useState<MediaItem[]>([]);
    const [alerts, setAlerts] = useState<AlertEvent[]>([
        { id: 1, time: '10:23:45', message: 'Detected Jumping', level: 'medium' },
        { id: 2, time: '09:12:30', message: 'No Motion for 1h', level: 'medium' },
    ]);

    const imgRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const recordingTimerRef = useRef<number | null>(null);

    // 🔥 幽灵连接彻底清理
    useEffect(() => {
        const currentImg = imgRef.current;
        return () => {
            if (currentImg) {
                currentImg.src = '';
            }
        };
    }, []);

    useEffect(() => {
        const init = async () => {
            const list = await getActiveCameras();
            setServerActiveCams(list);
            setIsActiveOnServer(list.includes(streamIndex));
        };
        init();
    }, [streamIndex]);

    useEffect(() => {
        setVideoError(false);
        setIsLoading(true);
        setRetryKey(Date.now());
    }, [cameraId]);

    const handleTogglePower = async () => {
        const currentlyActive = serverActiveCams.includes(streamIndex);
        let newList: number[];

        if (currentlyActive) {
            newList = serverActiveCams.filter(id => id !== streamIndex);
        } else {
            newList = [...serverActiveCams, streamIndex];
        }

        setServerActiveCams(newList);
        setIsActiveOnServer(newList.includes(streamIndex));

        if (!currentlyActive) {
            setIsLoading(true);
            setVideoError(false);
            setApiOffline(false);
            setRetryKey(Date.now());
        } else {
            setApiOffline(true);
        }

        await setActiveCameras(newList);
    };

    useEffect(() => {
        if (!apiOffline) {
            setVideoError(false);
            const timer = setTimeout(() => {
                setIsLoading(false);
            }, 1500);
            return () => clearTimeout(timer);
        } else {
            setIsLoading(true);
        }
    }, [apiOffline]);

    useEffect(() => {
        const fetchRealtimeData = async () => {
            const allStats = await fetchCameraStats();
            const camKey = isNaN(streamIndex) ? "0" : streamIndex.toString();
            const camData = allStats[camKey];

            if (camData && camData.stats) {
                const isOff = camData.stats.status === 'offline';
                setApiOffline(isOff);

                if (!isOff && videoError) {
                    setVideoError(false);
                    setIsLoading(true);
                    setRetryKey(prev => prev + 1);
                }

                setFps(camData.stats.fps || 0);
                const trackIds = Object.keys(camData.active_states || {});
                if (trackIds.length > 0) {
                    const mainObj = camData.active_states[trackIds[trackIds.length - 1]];
                    if (mainObj && mainObj.probs) {
                        let maxProb = 0;
                        let bestLabel = 'Rest';
                        Object.entries(mainObj.probs).forEach(([k, v]) => {
                            if (['Eat', 'Drink', 'Rest', 'Jump', 'Act'].includes(k)) {
                                if (v > maxProb) { maxProb = v; bestLabel = k; }
                            }
                        });
                        setCurrentStats({ behavior: bestLabel, confidence: Math.round(maxProb * 100) });
                    }
                }
            }
        };
        const interval = setInterval(fetchRealtimeData, 500);
        return () => clearInterval(interval);
    }, [streamIndex, videoError]);

    const handleSnapshot = () => { /* 实现省略 */ };
    const handleToggleRecording = () => { /* 实现省略 */ };
    const startRecording = () => { /* 实现省略 */ };
    const stopRecording = () => { /* 实现省略 */ };
    const downloadMedia = (item: MediaItem) => { /* 实现省略 */ };
    const deleteMedia = (id: string) => { /* 实现省略 */ };

    const isActuallyOffline = apiOffline || videoError;
    const isConnecting = !apiOffline && !videoError && isLoading;
    const isLive = !apiOffline && !videoError && !isLoading;

    return (
        <div className="h-full flex flex-col">
            <canvas ref={canvasRef} className="hidden" />

            <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-6 py-4 flex items-center justify-between shrink-0`}>
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${theme === 'dark' ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-primary-50 hover:bg-primary-100 text-primary-dark'}`}>
                        <ArrowLeft className="w-4 h-4" />
                        <span className="text-sm font-medium">{t('backToDashboard', language)}</span>
                    </button>

                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 gradient-primary rounded-xl flex items-center justify-center shadow-md">
                            <Camera className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h2 className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                                {t('liveMonitoring', language)} - {cameraName}
                            </h2>
                            <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                                {cameraId} · {isLive ? t('online', language) : t('offline', language)}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex-1 p-6 overflow-hidden">
                <div className="flex h-full gap-6">

                    <div className={`flex-1 flex flex-col ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border overflow-hidden relative`}>
                        <div className="flex-1 bg-black relative flex items-center justify-center overflow-hidden">

                            {!apiOffline && !videoError && (
                                <img
                                    ref={imgRef}
                                    key={retryKey}
                                    src={`${videoUrl}?k=${retryKey}`}
                                    alt="Live Stream"
                                    crossOrigin="anonymous"
                                    className={`w-full h-full object-contain ${isLoading ? 'hidden' : 'block'}`}
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
                                <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800 z-10">
                                    <Loader2 className="w-8 h-8 text-primary animate-spin mb-2" />
                                    <p className="text-xs text-gray-400 font-mono tracking-widest mt-2">LINKING TO KERNEL...</p>
                                </div>
                            )}

                            <div className="absolute top-6 left-6 right-6 flex items-center justify-between z-20 pointer-events-none">
                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={handleTogglePower}
                                        className={`pointer-events-auto flex items-center gap-2 px-4 py-2 rounded-xl backdrop-blur-md shadow-lg transition-all border
                                        ${isActiveOnServer
                                            ? 'bg-primary/80 border-primary/50 text-white hover:bg-primary'
                                            : 'bg-gray-900/80 border-gray-700 text-gray-300 hover:text-white hover:bg-gray-800'
                                        }`}
                                    >
                                        <Power className="w-4 h-4" />
                                        <span className="text-sm font-bold">{isActiveOnServer ? '后端推流 ON' : '后端推流 OFF'}</span>
                                    </button>

                                    {isLive && (
                                        <div className="flex items-center gap-2 bg-red-500/90 backdrop-blur-sm px-4 py-2 rounded-xl shadow-lg">
                                            <Circle className="w-3 h-3 text-white fill-white animate-pulse" />
                                            <span className="text-sm font-bold text-white">LIVE</span>
                                        </div>
                                    )}
                                    {isLive && (
                                        <div className="glass px-4 py-2 rounded-xl flex items-center gap-2">
                                            <Wifi className="w-4 h-4 text-primary" />
                                            <span className="text-sm font-medium text-white">1080p · {fps}fps</span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {isLive && (
                                <div className="absolute bottom-6 left-6 right-6 flex items-center justify-between z-20 pointer-events-none">
                                    <div className={`glass px-6 py-4 rounded-xl backdrop-blur-xl ${theme === 'dark' ? 'bg-gray-800/80' : 'bg-white/80'} flex items-center gap-8`}>
                                        <div>
                                            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>{t('behavior', language)}</p>
                                            <p className="text-xl font-bold text-primary">{t(currentStats.behavior, language)}</p>
                                        </div>
                                        <div className="text-right">
                                            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>{t('confidence', language)}</p>
                                            <div className="flex items-center gap-2">
                                                <Activity className="w-5 h-5 text-primary" />
                                                <span className="text-xl font-bold text-primary">{currentStats.confidence}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="w-96 flex flex-col gap-6">
                        <div className={`flex-1 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-4 overflow-hidden flex flex-col`}>
                            <div className="flex items-center gap-2 mb-4 pb-2 border-b border-gray-100 dark:border-gray-700 shrink-0">
                                <AlertTriangle className="w-5 h-5 text-orange-500" />
                                <h3 className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>异常提示</h3>
                            </div>
                            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
                                {alerts.length === 0 ? <div className="text-center text-gray-400 py-10 text-sm">暂无异常事件</div> : alerts.map(alert => (
                                    <div key={alert.id} className={`p-3 rounded-lg border-l-4 ${alert.level === 'high' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' : 'border-orange-400 bg-orange-50 dark:bg-orange-900/20'}`}>
                                        <div className="flex justify-between items-start mb-1"><span className={`text-xs font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>{alert.time}</span>{alert.level === 'high' && <span className="text-[10px] bg-red-500 text-white px-1.5 rounded">HIGH</span>}</div>
                                        <p className={`text-sm font-medium ${theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>{alert.message}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className={`flex-1 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-4 overflow-hidden flex flex-col`}>
                            <div className="flex items-center gap-2 mb-4 pb-2 border-b border-gray-100 dark:border-gray-700 shrink-0">
                                <FileImage className="w-5 h-5 text-blue-500" />
                                <h3 className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>快照 & 录像</h3>
                                <span className="ml-auto text-xs bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded-full text-gray-500">{mediaGallery.length}</span>
                            </div>
                            <div className="flex-1 overflow-y-auto pr-2 scrollbar-thin">
                                {mediaGallery.length === 0 ? <div className="h-full flex flex-col items-center justify-center text-gray-400 gap-2"><Camera className="w-8 h-8 opacity-20" /><p className="text-sm">点击左侧按钮抓拍或录制</p></div> : <div className="grid grid-cols-2 gap-3">
                                    {mediaGallery.map(item => (
                                        <div key={item.id} className="relative group aspect-video bg-black/10 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                                            {item.type === 'image' ? <img src={item.url} alt="snapshot" className="w-full h-full object-cover" /> : <video src={item.url} className="w-full h-full object-cover" />}
                                            <div className="absolute top-1 left-1 bg-black/60 px-1.5 rounded text-[10px] text-white flex items-center gap-1">{item.type === 'image' ? <FileImage className="w-3 h-3" /> : <FileVideo className="w-3 h-3" />}{item.type === 'video' && 'REC'}</div>
                                        </div>
                                    ))}
                                </div>}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}