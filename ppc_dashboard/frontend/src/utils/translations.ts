import { Language } from '../App';

export const translations = {
    'zh-CN': {
        dashboard: '监控仪表盘',
        petDataManagement: '宠物数据管理',
        monitoringAlerts: '监控告警',
        dataVisualization: '数据可视化',
        settings: '系统设置',

        // 🔥 后端行为映射
        Eat: '进食',
        Drink: '喝水',
        Rest: '休息',
        Jump: '跳跃',
        Act: '活动',

        // 通用状态
        normal: '正常',
        abnormal: '异常', // 虽然后端暂未触发，保留作为 Key
        online: '在线',
        offline: '离线',
        behavior: '当前行为',
        confidence: '置信度',

        // 界面文本
        totalCameras: '摄像头总数',
        activeCameras: '在线摄像头',
        abnormalEvents: '异常事件',
        detectionAccuracy: '识别准确率',
        quickMonitoring: '快速监控',
        viewAllCameras: '查看所有摄像头',
        liveMonitoring: '实时监控',
        backToDashboard: '返回仪表盘',
        fullscreen: '全屏',
        alertTimeline: '告警时间轴',
        realtimeAlerts: '实时告警',

        // 图表相关
        behaviorTrend: '行为趋势',
        behaviorDistribution: '行为分布',
        hourlyActivity: '每小时活动',
        monthlyTrend: '月度趋势',
        cameraPerformance: '摄像头性能',
        behaviorAnalysis: '行为分析',

        // 其他
        searchCameras: '搜索摄像头...',
        liveView: '实时画面',
        alertNotification: '告警通知',
        abnormalBehaviorDetected: '检测到异常行为',

        // 宠物管理
        addPet: '添加宠物',
        searchPets: '搜索宠物...',
        basicInfo: '基本信息',
        photos: '照片',
        videos: '视频',
        healthRecords: '健康记录',
        save: '保存',
        cancel: '取消',

        // 设置
        darkMode: '深色模式',
        lightMode: '浅色模式',
        fontSize: '字体大小',
        fontSizeSmall: '小',
        fontSizeMedium: '中',
        fontSizeLarge: '大',
        notifications: '通知',
        systemName: '智能宠物监控系统'
    },
    'zh-HK': {
        dashboard: '監控儀表板',
        petDataManagement: '寵物數據管理',
        monitoringAlerts: '監控告警',
        dataVisualization: '數據可視化',
        settings: '系統設置',

        Eat: '進食',
        Drink: '喝水',
        Rest: '休息',
        Jump: '跳躍',
        Act: '活動',

        normal: '正常',
        abnormal: '異常',
        online: '在線',
        offline: '離線',
        behavior: '當前行為',
        confidence: '置信度',

        totalCameras: '攝像頭總數',
        activeCameras: '在線攝像頭',
        abnormalEvents: '異常事件',
        detectionAccuracy: '識別準確率',
        quickMonitoring: '快速監控',
        viewAllCameras: '查看所有攝像頭',
        liveMonitoring: '實時監控',
        backToDashboard: '返回儀表板',
        fullscreen: '全屏',
        alertTimeline: '告警時間軸',
        realtimeAlerts: '實時告警',

        behaviorTrend: '行為趨勢',
        behaviorDistribution: '行為分佈',
        hourlyActivity: '每小時活動',
        monthlyTrend: '月度趨勢',
        cameraPerformance: '攝像頭性能',
        behaviorAnalysis: '行為分析',

        searchCameras: '搜索攝像頭...',
        liveView: '實時畫面',
        alertNotification: '告警通知',
        abnormalBehaviorDetected: '檢測到異常行為',

        addPet: '添加寵物',
        searchPets: '搜索寵物...',
        basicInfo: '基本信息',
        photos: '照片',
        videos: '影片',
        healthRecords: '健康記錄',
        save: '保存',
        cancel: '取消',

        darkMode: '深色模式',
        lightMode: '淺色模式',
        fontSize: '字體大小',
        fontSizeSmall: '小',
        fontSizeMedium: '中',
        fontSizeLarge: '大',
        notifications: '通知',
        systemName: '智能寵物監控系統'
    },
    'en': {
        dashboard: 'Dashboard',
        petDataManagement: 'Pet Data',
        monitoringAlerts: 'Alerts',
        dataVisualization: 'Analytics',
        settings: 'Settings',

        Eat: 'Eating',
        Drink: 'Drinking',
        Rest: 'Resting',
        Jump: 'Jumping',
        Act: 'Active',

        normal: 'Normal',
        abnormal: 'Abnormal',
        online: 'Online',
        offline: 'Offline',
        behavior: 'Behavior',
        confidence: 'Confidence',

        totalCameras: 'Total Cameras',
        activeCameras: 'Active Cameras',
        abnormalEvents: 'Abnormal Events',
        detectionAccuracy: 'Accuracy',
        quickMonitoring: 'Quick Monitoring',
        viewAllCameras: 'View All Cameras',
        liveMonitoring: 'Live Monitoring',
        backToDashboard: 'Back',
        fullscreen: 'Fullscreen',
        alertTimeline: 'Alert Timeline',
        realtimeAlerts: 'Real-time Alerts',

        behaviorTrend: 'Behavior Trend',
        behaviorDistribution: 'Distribution',
        hourlyActivity: 'Hourly Activity',
        monthlyTrend: 'Monthly Trend',
        cameraPerformance: 'Camera Performance',
        behaviorAnalysis: 'Analysis',

        searchCameras: 'Search cameras...',
        liveView: 'Live View',
        alertNotification: 'Alert Notification',
        abnormalBehaviorDetected: 'Abnormal behavior detected',

        addPet: 'Add Pet',
        searchPets: 'Search pets...',
        basicInfo: 'Basic Info',
        photos: 'Photos',
        videos: 'Videos',
        healthRecords: 'Health Records',
        save: 'Save',
        cancel: 'Cancel',

        darkMode: 'Dark Mode',
        lightMode: 'Light Mode',
        fontSize: 'Font Size',
        fontSizeSmall: 'Small',
        fontSizeMedium: 'Medium',
        fontSizeLarge: 'Large',
        notifications: 'Notifications',
        systemName: 'Smart Pet Monitor'
    }
};

export function t(key: string, language: Language): string {
    // @ts-ignore
    return translations[language]?.[key] || key;
}