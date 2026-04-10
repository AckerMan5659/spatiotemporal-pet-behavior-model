import React, { useState, useEffect, useRef } from 'react';
import { Camera, Activity, AlertTriangle, Target, TrendingUp, Video, Zap, Maximize2, Wifi, LayoutGrid, Info, PawPrint, WifiOff, Loader2, Cpu, BarChart2 } from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchCameraStats, getVideoFeedUrl, BackendStats, getActiveCameras, setActiveCameras } from '../services/api';

interface DashboardProps {
    language: Language;
    theme: Theme;
    onOpenLiveMonitoring: (cameraId: string, cameraName: string) => void;
    onNavigateToMonitoring?: () => void;
}

interface QuickCamera {
    id: string;
    name: string;
    status: 'online' | 'offline';
    currentBehavior: string;
    confidence: number;
    streamIndex: number;
    activeIds: string[];
}

interface HistoryPoint {
    time: string;
    camData: Record<string, Record<string, { behavior: string, probs: Record<string, number> }>>;
}

// 🏥 自訂攝影機 (病房) 名稱對應表
const CUSTOM_CAM_NAMES = [
    "中心病房1", "中心病房2", "隔離病房1", "隔離病房2",
    "貓病房1", "貓房1", "貓病房2", "貓房2"
];

export function Dashboard({ language, theme, onOpenLiveMonitoring, onNavigateToMonitoring }: DashboardProps) {
    const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
    const [quickCameras, setQuickCameras] = useState<QuickCamera[]>([]);
    const [activeCamCount, setActiveCamCount] = useState(0);

    const [selectedCamTab, setSelectedCamTab] = useState<string>('all');
    const [selectedTrackId, setSelectedTrackId] = useState<string>('all');

    const [historyData, setHistoryData] = useState<HistoryPoint[]>([]);
    const historyRef = useRef<HistoryPoint[]>([]);

    const sessionStatsRef = useRef({ totalConf: 0, count: 0 });
    const [sessionAccuracy, setSessionAccuracy] = useState(0);

    const [perfStats, setPerfStats] = useState({ yoloMs: 0, logicMs: 0 });
    const [serverActiveCams, setServerActiveCams] = useState<number[]>([]);

    const behaviorKeys = ['Eat', 'Drink', 'Rest', 'Jump', 'Act'];
    const behaviorColors: Record<string, string> = {
        Eat: '#83B5B5', Drink: '#9bc5c5', Rest: '#BFC5D5', Jump: '#E0BBE4', Act: '#BBD5D4',
    };

    useEffect(() => {
        const initCams = async () => {
            const activeList = await getActiveCameras();
            setServerActiveCams(activeList);
        };
        initCams();
    }, []);

    const handleToggleCamera = async (camIndex: number) => {
        const isActive = serverActiveCams.includes(camIndex);
        let newActiveList: number[];

        if (isActive) {
            newActiveList = serverActiveCams.filter(id => id !== camIndex);
        } else {
            newActiveList = [...serverActiveCams, camIndex];
        }

        setServerActiveCams(newActiveList);
        await setActiveCameras(newActiveList);
    };

    useEffect(() => {
        const loadStats = async () => {
            try {
                const data: BackendStats = await fetchCameraStats();
                const newCameras: QuickCamera[] = [];
                let activeCount = 0;

                const currentSnapshot: HistoryPoint = {
                    time: new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                    camData: {}
                };

                const camKeys = Object.keys(data).filter(k => !isNaN(Number(k)));

                let maxYolo = 0;
                let maxLogic = 0;
                camKeys.forEach(k => {
                    const s = data[k]?.stats;
                    if (s && s.status === 'online') {
                        if (s.yoloMs > maxYolo) maxYolo = s.yoloMs;
                        const currentLogic = (s.ruleMs || 0) + (s.actionMs || 0);
                        if (currentLogic > maxLogic) maxLogic = currentLogic;
                    }
                });

                setPerfStats({ yoloMs: maxYolo, logicMs: maxLogic });

                const numCamsToShow = Math.max(camKeys.length, 8);

                for (let i = 0; i < numCamsToShow; i++) {
                    const streamIndex = i;
                    const camKey = streamIndex.toString();
                    const camData = data[camKey];
                    const displayId = `CAM-${(i + 1).toString().padStart(2, '0')}`;

                    let mainBehavior = 'Rest';
                    let mainConfidence = 0;
                    let status: 'online' | 'offline' = 'offline';
                    let currentActiveIds: string[] = [];

                    currentSnapshot.camData[camKey] = {};

                    if (camData && camData.stats && camData.stats.status !== 'offline') {
                        status = 'online';
                        activeCount++;
                        const trackIds = Object.keys(camData.active_states || {});
                        currentActiveIds = trackIds;

                        trackIds.forEach(tid => {
                            const entity = camData.active_states[tid];
                            if (entity && entity.probs) {
                                let maxProb = 0;
                                let bestLabel = 'Rest';
                                Object.entries(entity.probs).forEach(([k, v]) => {
                                    if (behaviorKeys.includes(k)) {
                                        if (v > maxProb) { maxProb = v; bestLabel = k; }
                                    }
                                });
                                currentSnapshot.camData[camKey][tid] = { behavior: bestLabel, probs: entity.probs };
                                sessionStatsRef.current.totalConf += maxProb;
                                sessionStatsRef.current.count += 1;
                            }
                        });

                        if (trackIds.length > 0) {
                            const lastObj = currentSnapshot.camData[camKey][trackIds[trackIds.length-1]];
                            if (lastObj) {
                                mainBehavior = lastObj.behavior;
                                const probsValues = Object.values(lastObj.probs);
                                const maxP = probsValues.length > 0 ? Math.max(...probsValues) : 0;
                                mainConfidence = Math.round(maxP * 100);
                            }
                        }
                    }

                    // 🏥 套用自訂的病房名稱
                    const finalCamName = CUSTOM_CAM_NAMES[i] || `Camera ${i + 1}`;

                    newCameras.push({
                        id: displayId,
                        name: finalCamName,
                        status: status,
                        currentBehavior: mainBehavior,
                        confidence: mainConfidence,
                        streamIndex: streamIndex,
                        activeIds: currentActiveIds
                    });
                }

                setQuickCameras(newCameras);
                setActiveCamCount(activeCount);

                if (sessionStatsRef.current.count > 0) {
                    setSessionAccuracy((sessionStatsRef.current.totalConf / sessionStatsRef.current.count) * 100);
                }

                const newHistory = [...historyRef.current, currentSnapshot].slice(-20);
                historyRef.current = newHistory;
                setHistoryData(newHistory);
            } catch (error) {
                console.error("Dashboard Stats Error:", error);
            }
        };

        const interval = setInterval(loadStats, 1000);
        return () => clearInterval(interval);
    }, []);

    const getLineChartData = () => {
        if (!historyData || historyData.length === 0) return [];
        return historyData.map(point => {
            const item: any = { time: point.time };
            behaviorKeys.forEach(key => item[key] = 0);

            if (selectedCamTab === 'all') {
                Object.values(point.camData || {}).forEach(camTracks => {
                    Object.values(camTracks || {}).forEach(trackData => {
                        if (behaviorKeys.includes(trackData.behavior)) {
                            item[trackData.behavior] = (item[trackData.behavior] || 0) + 1;
                        }
                    });
                });
            } else {
                const camTracks = point.camData[selectedCamTab] || {};
                if (selectedTrackId === 'all') {
                    Object.values(camTracks).forEach(trackData => {
                        if (behaviorKeys.includes(trackData.behavior)) {
                            item[trackData.behavior] = (item[trackData.behavior] || 0) + 1;
                        }
                    });
                } else {
                    const trackData = camTracks[selectedTrackId];
                    if (trackData && trackData.probs) {
                        behaviorKeys.forEach(key => {
                            const val = trackData.probs[key] || 0;
                            item[key] = Number(val.toFixed(2));
                        });
                    }
                }
            }
            return item;
        });
    };

    const getPieChartData = () => {
        const counts: Record<string, number> = {};
        behaviorKeys.forEach(k => counts[k] = 0);
        const latestPoint = historyData[historyData.length - 1];
        if (!latestPoint) return [];

        if (selectedCamTab === 'all') {
            Object.values(latestPoint.camData || {}).forEach(camTracks => {
                Object.values(camTracks || {}).forEach(track => {
                    if (counts[track.behavior] !== undefined) counts[track.behavior]++;
                });
            });
        } else {
            const camTracks = latestPoint.camData[selectedCamTab] || {};
            Object.values(camTracks).forEach(track => {
                if (counts[track.behavior] !== undefined) counts[track.behavior]++;
            });
        }

        return Object.entries(counts).filter(([_, value]) => value > 0).map(([name, value]) => ({
            name: name, value: value, color: behaviorColors[name] || '#ccc'
        }));
    };

    const lineData = getLineChartData();
    const pieData = getPieChartData();

    const getCurrentActiveCount = () => {
        if (selectedCamTab === 'all') {
            return quickCameras.reduce((acc, cam) => acc + cam.activeIds.length, 0);
        } else {
            const cam = quickCameras.find(c => c.streamIndex.toString() === selectedCamTab);
            return cam ? cam.activeIds.length : 0;
        }
    };

    const statCards = [
        { icon: Camera, title: t('totalCameras', language), value: quickCameras.length.toString(), trend: 'Auto', iconBg: 'bg-primary' },
        { icon: Activity, title: t('activeCameras', language), value: activeCamCount.toString(), trend: 'Real-time', iconBg: 'bg-secondary' },
        { icon: AlertTriangle, title: t('abnormalEvents', language), value: '0', trend: 'Safe', iconBg: 'bg-warm' },
        { icon: Target, title: t('detectionAccuracy', language), value: `${sessionAccuracy.toFixed(1)}%`, trend: 'Avg', iconBg: 'bg-neutral' },
    ];

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {statCards.map((card, index) => <StatCard key={index} {...card} theme={theme} />)}
            </div>

            <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-6 card-hover`}>

                <div className="flex flex-row items-center justify-between gap-4 mb-6 flex-wrap">
                    <div className="flex items-center gap-3 shrink-0">
                        <div className="flex items-center gap-2 shrink-0">
                            <div className="w-9 h-9 bg-primary rounded-xl flex items-center justify-center shadow-md">
                                <Video className="w-5 h-5 text-white" />
                            </div>
                            <h3 className={`text-lg font-bold whitespace-nowrap ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                                {t('quickMonitoring', language)}
                            </h3>
                        </div>

                        <div className={`flex items-center gap-2 px-2 py-1 rounded-lg border shrink-0 ${theme === 'dark' ? 'bg-gray-900/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
                            <div className="flex items-center gap-1">
                                <Cpu className="w-3 h-3 text-blue-500" />
                                <span className={`text-[9px] font-medium whitespace-nowrap ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>YOLO:</span>
                                <span className={`text-[10px] font-bold font-mono ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{perfStats.yoloMs}ms</span>
                            </div>
                            <div className={`w-px h-2.5 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`}></div>
                            <div className="flex items-center gap-1">
                                <BarChart2 className="w-3 h-3 text-purple-500" />
                                <span className={`text-[9px] font-medium whitespace-nowrap ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Logic:</span>
                                <span className={`text-[10px] font-bold font-mono ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{perfStats.logicMs}ms</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 overflow-x-auto pb-1">
                        <span className={`text-xs font-bold shrink-0 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
                            後端推流控制:
                        </span>
                        {[0, 1, 2, 3, 4, 5, 6, 7].map((index) => {
                            const isOn = serverActiveCams.includes(index);
                            return (
                                <button
                                    key={index}
                                    onClick={() => handleToggleCamera(index)}
                                    className={`
                                        px-3 py-1 text-[10px] font-bold rounded-lg transition-all border whitespace-nowrap shrink-0
                                        ${isOn
                                        ? 'bg-primary text-white border-primary shadow-md hover:bg-primary/90'
                                        : theme === 'dark'
                                            ? 'bg-gray-900 text-gray-400 border-gray-700 hover:text-white'
                                            : 'bg-gray-100 text-gray-500 border-gray-200 hover:text-gray-800'
                                    }
                                    `}
                                >
                                    {CUSTOM_CAM_NAMES[index]} {isOn ? 'ON' : 'OFF'}
                                </button>
                            );
                        })}
                    </div>

                    <button onClick={onNavigateToMonitoring} className="flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all gradient-primary text-white hover:shadow-lg hover:scale-105 shrink-0 whitespace-nowrap">
                        <Zap className="w-3.5 h-3.5" />
                        <span className="text-xs font-medium">{t('viewAllCameras', language)}</span>
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {quickCameras.slice(0, 8).map((camera) => (
                        <QuickCameraCard
                            key={camera.id}
                            camera={camera}
                            language={language}
                            theme={theme}
                            isSelected={selectedCamera === camera.id}
                            onClick={() => setSelectedCamera(selectedCamera === camera.id ? null : camera.id)}
                            onOpenLiveMonitoring={onOpenLiveMonitoring}
                        />
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className={`lg:col-span-2 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-6 card-hover`}>
                    <div className="flex flex-col gap-4 mb-6">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-secondary rounded-xl flex items-center justify-center shadow-md">
                                    <TrendingUp className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h3 className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{t('behaviorTrend', language)}</h3>
                                    <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
                                        {(selectedCamTab !== 'all' && selectedTrackId !== 'all') ? `ID:${selectedTrackId} 實時機率 (0.0-1.0)` : `行為計數 (Count)`}
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div className={`flex p-1 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'} overflow-x-auto scrollbar-hide`}>
                            <button onClick={() => { setSelectedCamTab('all'); setSelectedTrackId('all'); }} className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all whitespace-nowrap flex items-center gap-1 ${selectedCamTab === 'all' ? 'bg-white text-primary shadow-sm' : theme === 'dark' ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-800'}`}>
                                <LayoutGrid className="w-3 h-3" /> All Cams
                            </button>
                            {quickCameras.map(cam => (
                                <button key={cam.id} onClick={() => { setSelectedCamTab(cam.streamIndex.toString()); setSelectedTrackId('all'); }} className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all whitespace-nowrap ${selectedCamTab === cam.streamIndex.toString() ? 'bg-white text-primary shadow-sm' : theme === 'dark' ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-800'}`}>{cam.name}</button>
                            ))}
                        </div>
                        {selectedCamTab !== 'all' && (
                            <div className={`flex items-center gap-2 overflow-x-auto pb-1`}>
                                <span className={`text-xs font-bold ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>Filter ID:</span>
                                <button onClick={() => setSelectedTrackId('all')} className={`px-2 py-1 rounded border text-xs transition-all ${selectedTrackId === 'all' ? 'bg-primary/10 border-primary text-primary' : theme === 'dark' ? 'border-gray-600 text-gray-400' : 'border-gray-300 text-gray-600'}`}>All IDs</button>
                                {(selectedCamTab !== 'all' ? quickCameras.find(c => c.streamIndex.toString() === selectedCamTab)?.activeIds || [] : []).map(id => (
                                    <button key={id} onClick={() => setSelectedTrackId(id)} className={`px-2 py-1 rounded border text-xs transition-all flex items-center gap-1 ${selectedTrackId === id ? 'bg-primary/10 border-primary text-primary' : theme === 'dark' ? 'border-gray-600 text-gray-400' : 'border-gray-300 text-gray-600'}`}><PawPrint className="w-3 h-3" /> {id}</button>
                                ))}
                            </div>
                        )}
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={lineData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} vertical={false} />
                            <XAxis dataKey="time" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} tick={{fontSize: 12}} interval="preserveStartEnd"/>
                            <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} domain={(selectedCamTab !== 'all' && selectedTrackId !== 'all') ? [0, 1] : [0, 'auto']} tickFormatter={(val) => (selectedCamTab !== 'all' && selectedTrackId !== 'all') ? val.toFixed(1) : Math.floor(val).toString()} />
                            <Tooltip contentStyle={{ backgroundColor: theme === 'dark' ? '#1f2937' : '#fff', border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`, borderRadius: '12px', color: theme === 'dark' ? '#fff' : '#000' }} labelStyle={{ fontWeight: 'bold', marginBottom: '0.5rem' }} formatter={(value: number) => [(selectedCamTab !== 'all' && selectedTrackId !== 'all') ? value.toFixed(2) : value, "" ]} />
                            <Legend wrapperStyle={{ paddingTop: '10px' }}/>
                            {behaviorKeys.map(key => <Line key={key} type="monotone" dataKey={key} name={t(key, language)} stroke={behaviorColors[key]} strokeWidth={2.5} dot={false} activeDot={{ r: 6 }} animationDuration={300} />)}
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-6 card-hover`}>
                    <div className="flex items-center gap-3 mb-6">
                        <div className="w-10 h-10 bg-neutral rounded-xl flex items-center justify-center shadow-md"><Target className="w-5 h-5 text-white" /></div>
                        <div>
                            <h3 className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{t('behaviorDistribution', language)}</h3>
                            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'} flex items-center gap-1`}><Info className="w-3 h-3" />{selectedCamTab === 'all' ? '所有在線攝影機統計' : `${quickCameras.find(c => c.streamIndex.toString() === selectedCamTab)?.name || 'Selected'} 攝影機統計`}</p>
                        </div>
                    </div>
                    <div className="relative">
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                                    {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} strokeWidth={0} />)}
                                </Pie>
                                <Tooltip formatter={(value: number, name: string) => [value, t(name, language)]} contentStyle={{ backgroundColor: theme === 'dark' ? '#1f2937' : '#fff', border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`, borderRadius: '12px', color: theme === 'dark' ? '#fff' : '#000' }} />
                                <Legend formatter={(value) => t(value, language)} verticalAlign="bottom" height={36} />
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none mb-8">
                            <p className={`text-3xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{getCurrentActiveCount()}</p>
                            <p className={`text-xs font-medium ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>Active IDs</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function StatCard({ icon: Icon, title, value, trend, iconBg, theme }: any) {
    return (
        <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl shadow-lg border p-6 card-hover relative overflow-hidden`}>
            <div className="flex items-start justify-between relative z-10">
                <div>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>{title}</p>
                    <p className={`text-3xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{value}</p>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'} mt-2`}>{trend}</p>
                </div>
                <div className={`w-14 h-14 ${iconBg} rounded-xl flex items-center justify-center shadow-md`}>
                    <Icon className="w-7 h-7 text-white" />
                </div>
            </div>
        </div>
    );
}

function QuickCameraCard({ camera, language, theme, isSelected, onClick, onOpenLiveMonitoring }: any) {
    const [videoError, setVideoError] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [retryKey, setRetryKey] = useState(0);
    const imgRef = useRef<HTMLImageElement>(null);
    const videoFeedUrl = getVideoFeedUrl(camera.streamIndex);

    const handleDoubleClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        onOpenLiveMonitoring(camera.id, camera.name);
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
        if (camera.status === 'online') {
            setVideoError(false);
            const timer = setTimeout(() => {
                setIsLoading(false);
            }, 1500);
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
            onClick={onClick}
            onDoubleClick={handleDoubleClick}
            className={`
        relative rounded-xl overflow-hidden cursor-pointer transition-all duration-300
        ${isSelected ? 'ring-4 ring-primary shadow-xl shadow-primary/20 scale-105' : 'hover:scale-102 hover:shadow-lg'}
        ${theme === 'dark' ? 'bg-gray-700/50' : 'bg-white/90'}
        glass
      `}
        >
            <div className="relative bg-black aspect-video overflow-hidden group">
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
                        <p className="text-xs text-gray-400">Connecting...</p>
                    </div>
                )}
                {isLive && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-30">
                        <div className="absolute top-2 left-2 flex items-center gap-1 bg-red-500/90 backdrop-blur-sm px-2 py-1 rounded-lg shadow-md">
                            <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                            <span className="text-xs font-bold text-white">LIVE</span>
                        </div>
                        <div className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm px-3 py-1 rounded-lg z-10">
                            <span className="text-xs font-medium text-white">{camera.name}</span>
                        </div>
                        {camera.activeIds.length > 0 && (
                            <div className="absolute bottom-2 left-2 glass px-2 py-1 rounded-lg flex items-center gap-1">
                                <PawPrint className="w-3 h-3 text-white" />
                                <span className="text-xs font-medium text-white">{camera.activeIds.length} Pets</span>
                            </div>
                        )}
                        {isSelected && (
                            <div className="absolute inset-0 bg-primary/20 backdrop-blur-sm flex items-center justify-center">
                                <Maximize2 className="w-8 h-8 text-white animate-pulse" />
                            </div>
                        )}
                    </div>
                )}
            </div>
            <div className="p-3">
                <div className="flex items-center justify-between mb-2">
                    <h4 className={`font-medium text-sm ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>{camera.name}</h4>
                    {isActuallyOffline ? <WifiOff className="w-3 h-3 text-gray-400" /> : <Wifi className="w-3 h-3 text-primary" />}
                </div>
                {!isActuallyOffline ? (
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <span className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>{t('behavior', language)}:</span>
                            <span className="text-xs font-medium px-2 py-0.5 rounded-lg bg-primary text-white shadow-sm">{t(camera.currentBehavior, language)}</span>
                        </div>
                        <div>
                            <div className="flex items-center justify-between mb-1">
                                <span className={`text-xs font-medium ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>{camera.confidence}%</span>
                            </div>
                            <div className={`w-full rounded-full h-2 ${theme === 'dark' ? 'bg-gray-600' : 'bg-gray-200'}`}>
                                <div className={`h-2 rounded-full bg-primary transition-all`} style={{ width: `${camera.confidence}%` }} />
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="h-12 flex items-center justify-center"><span className="text-xs text-gray-400">Signal Lost</span></div>
                )}
            </div>
        </div>
    );
}