import React, { useState } from 'react';
import { Dashboard } from './components/Dashboard';
import { PetDataManagement } from './components/PetDataManagement';
import { MonitoringAlerts } from './components/MonitoringAlerts';
import { LiveMonitoring } from './components/LiveMonitoring';
import { Settings } from './components/Settings';
import { DataVisualization } from './components/DataVisualization';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';

export type Page = 'dashboard' | 'petData' | 'monitoring' | 'settings';
export type Language = 'zh-CN' | 'zh-HK' | 'en';
export type FontSize = 'small' | 'medium' | 'large';
export type Theme = 'light' | 'dark';

export default function App() {
    const [currentPage, setCurrentPage] = useState<Page>('dashboard');
    const [language, setLanguage] = useState<Language>('zh-CN');
    const [fontSize, setFontSize] = useState<FontSize>('medium');
    const [theme, setTheme] = useState<Theme>('light');
    const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
    const [liveMonitoringCamera, setLiveMonitoringCamera] = useState<{
        cameraId: string;
        cameraName: string;
    } | null>(null);
    const [showDataVisualization, setShowDataVisualization] = useState(false);

    // 打开直播监控的回调函数
    const handleOpenLiveMonitoring = (cameraId: string, cameraName: string) => {
        setLiveMonitoringCamera({ cameraId, cameraName });
    };

    const handleCloseLiveMonitoring = () => {
        setLiveMonitoringCamera(null);
    };

    const handleOpenDataVisualization = () => {
        setShowDataVisualization(true);
    };

    const handleCloseDataVisualization = () => {
        setShowDataVisualization(false);
    };

    const renderPage = () => {
        // 1. 直播页面优先
        if (liveMonitoringCamera) {
            return (
                <LiveMonitoring
                    language={language}
                    theme={theme}
                    cameraId={liveMonitoringCamera.cameraId}
                    cameraName={liveMonitoringCamera.cameraName}
                    onBack={handleCloseLiveMonitoring}
                />
            );
        }

        // 2. 数据大屏次之
        if (showDataVisualization) {
            return (
                <DataVisualization
                    language={language}
                    theme={theme}
                    onBack={handleCloseDataVisualization}
                />
            );
        }

        // 3. 常规页面路由
        switch (currentPage) {
            case 'dashboard':
                return (
                    <Dashboard
                        language={language}
                        theme={theme}
                        onOpenLiveMonitoring={handleOpenLiveMonitoring}
                        onNavigateToMonitoring={() => setCurrentPage('monitoring')}
                    />
                );
            case 'petData':
                return <PetDataManagement language={language} theme={theme} />;
            case 'monitoring':
                // 🔥 修改：将 onOpenLiveMonitoring 传递给 MonitoringAlerts
                return (
                    <MonitoringAlerts
                        language={language}
                        theme={theme}
                        onOpenLiveMonitoring={handleOpenLiveMonitoring}
                    />
                );
            case 'settings':
                return <Settings language={language} theme={theme} />;
            default:
                return (
                    <Dashboard
                        language={language}
                        theme={theme}
                        onOpenLiveMonitoring={handleOpenLiveMonitoring}
                    />
                );
        }
    };

    const fontSizeClasses = {
        small: 'text-sm',
        medium: 'text-base',
        large: 'text-lg'
    };

    return (
        <div className={`flex h-screen overflow-hidden ${fontSizeClasses[fontSize]} ${theme === 'dark' ? 'dark' : ''}`}>
            {/* 侧边栏：非全屏模式下显示 */}
            {!liveMonitoringCamera && !showDataVisualization && (
                <Sidebar
                    currentPage={currentPage}
                    setCurrentPage={setCurrentPage}
                    language={language}
                    theme={theme}
                    isCollapsed={isCollapsed}
                    setIsCollapsed={setIsCollapsed}
                    onOpenDataVisualization={handleOpenDataVisualization}
                />
            )}

            <div className={`flex-1 flex flex-col overflow-hidden ${theme === 'dark' ? 'bg-gray-900' : 'bg-gradient-to-br from-accent-light via-white to-secondary-light/30'}`}>
                {/* 顶部栏：非全屏模式下显示 */}
                {!liveMonitoringCamera && !showDataVisualization && (
                    <TopBar
                        language={language}
                        setLanguage={setLanguage}
                        fontSize={fontSize}
                        setFontSize={setFontSize}
                        theme={theme}
                        setTheme={setTheme}
                    />
                )}

                <main
                    className={`flex-1 overflow-auto ${(liveMonitoringCamera || showDataVisualization) ? '' : 'p-6'} ${theme === 'dark' ? 'bg-gray-900' : 'bg-white'}`}
                >
                    {renderPage()}
                </main>
            </div>
        </div>
    );
}