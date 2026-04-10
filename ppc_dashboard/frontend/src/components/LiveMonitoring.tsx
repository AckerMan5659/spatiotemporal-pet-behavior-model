import React, { useState, useEffect, useRef } from 'react';
import {
    ArrowLeft, Camera, Circle, Activity, Wifi, Settings,
    AlertTriangle, Power, WifiOff, Loader2, Clock, Search
} from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';
import { getVideoFeedUrl, fetchCameraStats, getActiveCameras, setActiveCameras, API_BASE_URL } from '../services/api';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface LiveMonitoringProps {
    language: Language;
    theme: Theme;
    cameraId: string;
    cameraName: string;
    onBack: () => void;
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

    const [apiOffline, setApiOffline] = useState(true);
    const [isLoading, setIsLoading] = useState(true);
    const [videoError, setVideoError] = useState(false);
    const [retryKey, setRetryKey] = useState(0);

    const [serverActiveCams, setServerActiveCams] = useState<number[]>([]);
    const [isActiveOnServer, setIsActiveOnServer] = useState(false);

    const [alerts, setAlerts] = useState<AlertEvent[]>([
        { id: 1, time: '10:23:45', message: 'Detected Jumping', level: 'medium' },
        { id: 2, time: '09:12:30', message: 'No Motion for 1h', level: 'medium' },
    ]);

    const imgRef = useRef<HTMLImageElement>(null);

    const [historicalStats, setHistoricalStats] = useState<Record<string, number>>({});
    const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | 'custom'>('1h');
    const [customStart, setCustomStart] = useState('');
    const [customEnd, setCustomEnd] = useState('');
    const [isFetchingHistory, setIsFetchingHistory] = useState(false);

    const behaviorColors: Record<string, string> = {
        Eat: '#83B5B5', Drink: '#9bc5c5', Rest: '#BFC5D5', Jump: '#E0BBE4', Act: '#BBD5D4',
    };

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

    const fetchHistoricalBehavior = async () => {
        setIsFetchingHistory(true);
        try {
            let start = '';
            let end = '';
            const now = new Date();

            if (timeRange === '1h') {
                start = new Date(now.getTime() - 60 * 60 * 1000).toISOString().replace('T', ' ').substring(0, 19);
            } else if (timeRange === '24h') {
                start = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString().replace('T', ' ').substring(0, 19);
            } else if (timeRange === '7d') {
                start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString().replace('T', ' ').substring(0, 19);
            } else if (timeRange === 'custom') {
                if (customStart) {
                    start = customStart.replace('T', ' ');
                    if (start.split(':').length === 2) start += ':00';
                }
                if (customEnd) {
                    end = customEnd.replace('T', ' ');
                    if (end.split(':').length === 2) end += ':59';
                }
            }

            let url = `${API_BASE_URL}/api/behavior_logs?cam_id=${streamIndex}`;
            if (start) url += `&start=${encodeURIComponent(start)}`;
            if (end) url += `&end=${encodeURIComponent(end)}`;

            const res = await fetch(url);
            const data = await res.json();
            if (data.success && data.stats) {
                setHistoricalStats(data.stats);
            } else {
                setHistoricalStats({});
            }
        } catch (e) {
            console.error('Failed to fetch behavior history:', e);
            setHistoricalStats({});
        } finally {
            setIsFetchingHistory(false);
        }
    };

    useEffect(() => {
        if (timeRange !== 'custom') {
            fetchHistoricalBehavior();
        }
        
        let interval: any;
        if (timeRange === '1h') {
            interval = setInterval(fetchHistoricalBehavior, 10000);
        }
        return () => clearInterval(interval);
    }, [timeRange, streamIndex]);

    const pieData = Object.entries(historicalStats)
        .filter(([_, value]) => value > 0)
        .map(([name, value]) => ({
            name: name,
            value: value,
            color: behaviorColors[name] || '#ccc'
        }));

    const isActuallyOffline = apiOffline || videoError;
    const isConnecting = !apiOffline && !videoError && isLoading;
    const isLive = !apiOffline && !videoError && !isLoading;

    return (
        <div className="h-full flex flex-col">

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
                        <div className={`h-[30%] min-h-[200px] ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-4 flex flex-col`}>
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

                        {/* 歷史行為統計分析圖 */}
                        <div className={`flex-1 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-4 flex flex-col overflow-hidden`}>
                            
                            <div className="flex items-center justify-between mb-2 pb-2 border-b border-gray-100 dark:border-gray-700 shrink-0 relative z-20">
                                <div className="flex items-center gap-2">
                                    <Clock className="w-5 h-5 text-blue-500" />
                                    <h3 className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>歷史行為統計</h3>
                                </div>
                                
                                <div className="relative">
                                    <select 
                                        value={timeRange} 
                                        onChange={(e: any) => setTimeRange(e.target.value)}
                                        className={`text-xs p-1.5 rounded border outline-none focus:ring-1 focus:ring-primary cursor-pointer ${theme === 'dark' ? 'bg-gray-900 border-gray-600 text-white' : 'bg-gray-50 border-gray-300 text-gray-800'}`}
                                    >
                                        <option value="1h">近 1 小時</option>
                                        <option value="24h">近 24 小時</option>
                                        <option value="7d">近 7 天</option>
                                        <option value="custom">自訂時間</option>
                                    </select>

                                    {timeRange === 'custom' && (
                                        <div className={`absolute top-full right-0 mt-2 p-3 rounded-xl border shadow-2xl flex flex-col gap-3 w-56 z-50 ${theme === 'dark' ? 'bg-gray-800 border-gray-600' : 'bg-white border-gray-200'}`}>
                                            <div className="flex flex-col gap-1">
                                                <label className={`text-[10px] font-bold ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>開始時間</label>
                                                <input 
                                                    type="datetime-local" 
                                                    step="1"
                                                    value={customStart} 
                                                    onChange={(e) => setCustomStart(e.target.value)} 
                                                    className={`w-full text-xs p-1.5 border rounded outline-none focus:ring-1 focus:ring-primary ${theme === 'dark' ? 'bg-gray-900 border-gray-600 text-white [color-scheme:dark]' : 'bg-white border-gray-300 text-gray-800'}`} 
                                                />
                                            </div>
                                            <div className="flex flex-col gap-1">
                                                <label className={`text-[10px] font-bold ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>結束時間</label>
                                                <input 
                                                    type="datetime-local" 
                                                    step="1"
                                                    value={customEnd} 
                                                    onChange={(e) => setCustomEnd(e.target.value)} 
                                                    className={`w-full text-xs p-1.5 border rounded outline-none focus:ring-1 focus:ring-primary ${theme === 'dark' ? 'bg-gray-900 border-gray-600 text-white [color-scheme:dark]' : 'bg-white border-gray-300 text-gray-800'}`} 
                                                />
                                            </div>
                                            <button 
                                                onClick={fetchHistoricalBehavior}
                                                disabled={!customStart || !customEnd || isFetchingHistory}
                                                className="w-full mt-1 py-1.5 rounded-lg bg-primary hover:bg-primary/90 text-white text-xs font-bold transition-all disabled:opacity-50 shadow-md flex items-center justify-center gap-1"
                                            >
                                                {isFetchingHistory ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                                                套用查詢
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* 🔥 餅圖區域：增加了外層容器的高度與餅圖的內外半徑 */}
                            <div className="flex-1 w-full relative min-h-[180px] z-10 mt-2">
                                {isFetchingHistory ? (
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <Loader2 className="w-6 h-6 animate-spin text-primary" />
                                    </div>
                                ) : pieData.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie 
                                                data={pieData} 
                                                cx="50%" cy="50%" 
                                                innerRadius={65}   /* 半徑調大 */
                                                outerRadius={95}   /* 半徑調大 */
                                                paddingAngle={5} dataKey="value"
                                                animationDuration={500}
                                            >
                                                {pieData.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} strokeWidth={0} />
                                                ))}
                                            </Pie>
                                            <Tooltip 
                                                formatter={(value: number, name: string) => [`${value} 幀/秒`, t(name, language)]}
                                                contentStyle={{ backgroundColor: theme === 'dark' ? '#1f2937' : '#fff', border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`, borderRadius: '12px', color: theme === 'dark' ? '#fff' : '#000', fontSize: '12px', padding: '4px 8px' }}
                                            />
                                            <Legend formatter={(value) => t(value, language)} verticalAlign="bottom" height={24} iconSize={10} wrapperStyle={{ fontSize: '12px' }} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>無該時段之活動紀錄</p>
                                    </div>
                                )}
                            </div>
                            <p className="text-[9px] text-gray-400 text-center mt-3 shrink-0 relative z-10">
                                * 若畫面存在多目標，已取比例最高之主導行為。
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}