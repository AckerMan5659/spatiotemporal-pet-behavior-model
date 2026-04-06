import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, Activity, Video, Clock, Target, Zap, Calendar, Award, X } from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';
import { fetchCameraStats, BackendStats } from '../services/api';

interface DataVisualizationProps {
    language: Language;
    theme: Theme;
    onBack: () => void;
}

export function DataVisualization({ language, theme, onBack }: DataVisualizationProps) {
    const [currentTime, setCurrentTime] = useState(new Date());
    const [animatedValue, setAnimatedValue] = useState(0);

    const [realStats, setRealStats] = useState({
        totalRecords: 0,
        avgAccuracy: 0,
        abnormalCount: 0
    });

    const [monitorDuration, setMonitorDuration] = useState('0天 0小时 0分 0秒');
    const [behaviorData, setBehaviorData] = useState<any[]>([]);
    const [radarData, setRadarData] = useState<any[]>([]);
    const [cameraPerfData, setCameraPerfData] = useState<any[]>([]);

    const sessionStatsRef = useRef({
        totalConfidenceSum: 0,
        sampleCount: 0
    });

    useEffect(() => {
        const storedStart = localStorage.getItem('sys_monitor_start_time');
        let startTime: number;

        if (storedStart) {
            startTime = parseInt(storedStart, 10);
        } else {
            startTime = Date.now();
            localStorage.setItem('sys_monitor_start_time', startTime.toString());
        }

        const timer = setInterval(() => {
            setCurrentTime(new Date());
            const now = Date.now();
            const diff = now - startTime;

            const days = Math.floor(diff / (1000 * 60 * 60 * 24));
            const hours = Math.floor((diff / (1000 * 60 * 60)) % 24);
            const minutes = Math.floor((diff / (1000 * 60)) % 60);
            const seconds = Math.floor((diff / 1000) % 60);

            setMonitorDuration(`${days}天 ${hours}小时 ${minutes}分 ${seconds}秒`);
        }, 1000);

        return () => clearInterval(timer);
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            setAnimatedValue(prev => (prev + 1) % 100);
        }, 50);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const loadRealData = async () => {
            const data: BackendStats = await fetchCameraStats();

            let currentPollTotalRecs = 0;
            const counts = { Eat: 0, Drink: 0, Rest: 0, Jump: 0, Act: 0 };
            const avgProbs = { Eat: 0, Drink: 0, Rest: 0, Jump: 0, Act: 0 };
            let probSamples = 0;
            const cameraPerf: any[] = [];

            Object.keys(data).forEach(key => {
                const cam = data[key];
                // 🔥 修复这里的 0 拼接逻辑以支持多于 9 路流
                const displayId = `CAM-${(parseInt(key) + 1).toString().padStart(2, '0')}`;

                const activeIds = Object.keys(cam.active_states || {});

                activeIds.forEach(trackId => {
                    const entity = cam.active_states[trackId];
                    if (entity && entity.probs) {
                        const probsValues = Object.values(entity.probs) as number[];
                        if (probsValues.length > 0) {
                            const maxProb = Math.max(...probsValues);
                            sessionStatsRef.current.totalConfidenceSum += maxProb;
                            sessionStatsRef.current.sampleCount++;
                        }
                    }
                });

                if (cam.logs) {
                    currentPollTotalRecs += cam.logs.length;
                    cam.logs.forEach((log: any) => {
                        let maxP = 0;
                        let maxK = 'Rest';
                        if (log.probs) {
                            probSamples++;
                            Object.entries(log.probs).forEach(([k, v]: [string, any]) => {
                                if (counts.hasOwnProperty(k)) {
                                    avgProbs[k as keyof typeof avgProbs] += v;
                                    if (v > maxP) { maxP = v; maxK = k; }
                                }
                            });
                        }
                        if (counts.hasOwnProperty(maxK)) counts[maxK as keyof typeof counts]++;
                    });
                }

                cameraPerf.push({
                    camera: displayId,
                    uptime: cam.stats.status === 'online' ? 100 : 0,
                    fps: cam.stats.fps,
                    detections: activeIds.length
                });
            });

            let sessionAvgAccuracy = 0;
            if (sessionStatsRef.current.sampleCount > 0) {
                sessionAvgAccuracy = (sessionStatsRef.current.totalConfidenceSum / sessionStatsRef.current.sampleCount) * 100;
            }

            setRealStats({
                totalRecords: currentPollTotalRecs * 10,
                avgAccuracy: Number(sessionAvgAccuracy.toFixed(1)),
                abnormalCount: 0
            });

            setBehaviorData([
                { name: t('Rest', language), value: counts.Rest, color: '#BFC5D5' },
                { name: t('Act', language), value: counts.Act, color: '#BBD5D4' },
                { name: t('Eat', language), value: counts.Eat, color: '#83B5B5' },
                { name: t('Drink', language), value: counts.Drink, color: '#9bc5c5' },
                { name: t('Jump', language), value: counts.Jump, color: '#E0BBE4' },
            ].filter(d => d.value > 0));

            if (probSamples > 0) {
                setRadarData([
                    { behavior: t('Eat', language), current: Math.round((avgProbs.Eat / probSamples) * 100), full: 100 },
                    { behavior: t('Rest', language), current: Math.round((avgProbs.Rest / probSamples) * 100), full: 100 },
                    { behavior: t('Act', language), current: Math.round((avgProbs.Act / probSamples) * 100), full: 100 },
                    { behavior: t('Drink', language), current: Math.round((avgProbs.Drink / probSamples) * 100), full: 100 },
                    { behavior: t('Jump', language), current: Math.round((avgProbs.Jump / probSamples) * 100), full: 100 },
                ]);
            }
            setCameraPerfData(cameraPerf);
        };

        loadRealData();
        const interval = setInterval(loadRealData, 1000);
        return () => clearInterval(interval);
    }, [language]);

    const monthlyData = [
        { month: '1月', records: 1245, accuracy: 94.5 },
        { month: '2月', records: 1389, accuracy: 95.2 },
        { month: '3月', records: 1567, accuracy: 93.8 },
    ];

    return (
        <div className={`min-h-screen ${theme === 'dark' ? 'bg-gray-900' : 'bg-gradient-to-br from-blue-50 via-white to-purple-50'} p-8 relative`}>
            <button onClick={onBack} className={`fixed top-4 left-4 z-50 px-3 py-1.5 rounded-lg text-xs flex items-center gap-1.5 backdrop-blur-sm shadow-md ${theme === 'dark' ? 'bg-gray-800/80 text-gray-300' : 'bg-white/80 text-gray-600'}`}>
                <X className="w-3 h-3" />
                <span className="font-medium">返回</span>
            </button>

            <div className="mb-8">
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h1 className={`text-4xl font-bold mb-2 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent`}>
                            {t('dataVisualization', language)}
                        </h1>
                        <p className={`text-lg ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                            实时数据分析控制台
                        </p>
                    </div>
                    <div className={`text-right ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                        <div className="text-3xl font-bold font-mono">
                            {currentTime.toLocaleTimeString('zh-CN', { hour12: false })}
                        </div>
                        <div className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                            {currentTime.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })}
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <StatCard
                        icon={<Video className="w-8 h-8" />}
                        title={t('totalRecords', language)}
                        value={realStats.totalRecords.toLocaleString()}
                        suffix={t('records', language)}
                        gradient="from-blue-500 to-blue-700"
                        theme={theme}
                        animated={animatedValue}
                    />

                    <StatCard
                        icon={<Clock className="w-8 h-8" />}
                        title={t('totalMonitoringTime', language)}
                        value={monitorDuration}
                        suffix=""
                        gradient="from-purple-500 to-purple-700"
                        theme={theme}
                        animated={animatedValue}
                        valueSize="text-xl"
                    />

                    <StatCard
                        icon={<Target className="w-8 h-8" />}
                        title={t('detectionAccuracy', language)}
                        value={realStats.avgAccuracy}
                        suffix="%"
                        gradient="from-green-500 to-green-700"
                        theme={theme}
                        animated={animatedValue}
                    />

                    <StatCard
                        icon={<Zap className="w-8 h-8" />}
                        title={t('abnormalEvents', language)}
                        value={realStats.abnormalCount}
                        suffix=""
                        gradient="from-red-500 to-red-700"
                        theme={theme}
                        animated={animatedValue}
                    />
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <ChartCard title={`${t('behaviorDistribution', language)} (实时)`} theme={theme}>
                    {behaviorData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={behaviorData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                    outerRadius={100}
                                    dataKey="value"
                                >
                                    {behaviorData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip contentStyle={{ backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff', border: 'none', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-[300px] flex items-center justify-center text-gray-400">
                            等待数据同步...
                        </div>
                    )}
                </ChartCard>

                <ChartCard title={`${t('behaviorAnalysis', language)} (平均置信度)`} theme={theme}>
                    {radarData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={300}>
                            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                                <PolarGrid stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                                <PolarAngleAxis dataKey="behavior" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                                <PolarRadiusAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                                <Radar name="Avg Confidence" dataKey="current" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                                <Legend />
                                <Tooltip contentStyle={{ backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff', border: 'none', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                            </RadarChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-[300px] flex items-center justify-center text-gray-400">
                            等待数据同步...
                        </div>
                    )}
                </ChartCard>

                <ChartCard title={t('cameraPerformance', language)} theme={theme}>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={cameraPerfData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                            <XAxis dataKey="camera" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                            <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                            <Tooltip contentStyle={{ backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff', border: 'none', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                            <Legend />
                            <Bar dataKey="fps" fill="#10b981" name="FPS" radius={[8, 8, 0, 0]} />
                            <Bar dataKey="detections" fill="#3b82f6" name="Active Objects" radius={[8, 8, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>

                <ChartCard title={`${t('monthlyTrend', language)} (示例数据)`} theme={theme}>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={monthlyData}>
                            <defs>
                                <linearGradient id="colorRecords" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                            <XAxis dataKey="month" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                            <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                            <Tooltip />
                            <Area type="monotone" dataKey="records" stroke="#3b82f6" fillOpacity={1} fill="url(#colorRecords)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </ChartCard>
            </div>
        </div>
    );
}

function StatCard({ icon, title, value, suffix, gradient, theme, animated, valueSize }: any) {
    return (
        <div
            className={`
        ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'} 
        rounded-2xl p-6 border shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1
        relative overflow-hidden group
      `}
        >
            <div
                className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}
                style={{ transform: `translateX(${animated}%)` }}
            />
            <div className="relative z-10">
                <div className={`w-14 h-14 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center mb-4 text-white shadow-lg`}>
                    {icon}
                </div>
                <p className={`text-sm mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{title}</p>
                <div className="flex items-baseline gap-2">
                    <p className={`${valueSize || 'text-3xl'} font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{value}</p>
                    {suffix && <span className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{suffix}</span>}
                </div>
            </div>
        </div>
    );
}

function ChartCard({ title, theme, children }: any) {
    return (
        <div className={`
      ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'} 
      rounded-2xl p-6 border shadow-lg
    `}>
            <h3 className={`text-xl font-bold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                {title}
            </h3>
            {children}
        </div>
    );
}