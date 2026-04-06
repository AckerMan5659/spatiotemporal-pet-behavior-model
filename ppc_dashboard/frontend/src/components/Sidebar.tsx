import React from 'react';
import { LayoutDashboard, Database, Video, PawPrint, ChevronLeft, ChevronRight, Settings, BarChart3 } from 'lucide-react';
import { Language, Page, Theme } from '../App';
import { t } from '../utils/translations';

interface SidebarProps {
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
  language: Language;
  theme: Theme;
  isCollapsed: boolean;
  setIsCollapsed: (collapsed: boolean) => void;
  onOpenDataVisualization: () => void;
}

export function Sidebar({ currentPage, setCurrentPage, language, theme, isCollapsed, setIsCollapsed, onOpenDataVisualization }: SidebarProps) {
  const menuItems = [
    {
      id: 'dashboard' as Page,
      icon: LayoutDashboard,
      label: t('dashboard', language),
    },
    {
      id: 'petData' as Page,
      icon: Database,
      label: t('petDataManagement', language),
    },
    {
      id: 'monitoring' as Page,
      icon: Video,
      label: t('monitoringAlerts', language),
    },
    {
      id: 'dataVisualization' as Page,
      icon: BarChart3,
      label: t('dataVisualization', language),
    }
  ];

  return (
    <aside className={`${isCollapsed ? 'w-20' : 'w-64'} ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} flex flex-col shadow-xl border-r transition-all duration-300`}>
      {/* Logo Area */}
      <div className={`p-6 border-b ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'} ${isCollapsed ? 'px-4' : ''}`}>
        <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
          <div className="w-12 h-12 gradient-primary rounded-xl flex items-center justify-center shadow-lg relative overflow-hidden flex-shrink-0">
            <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
            <PawPrint className="w-7 h-7 text-white relative z-10" />
          </div>
          {!isCollapsed && (
            <div className="transition-opacity duration-300">
              <h1 className={`font-bold leading-tight ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                智能宠物监控
              </h1>
              <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>行为推理平台</p>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          const isDataViz = item.id === 'dataVisualization';
          
          return (
            <button
              key={item.id}
              onClick={() => {
                if (isDataViz) {
                  onOpenDataVisualization();
                } else {
                  setCurrentPage(item.id);
                }
              }}
              className={`
                w-full flex items-center gap-3 px-4 py-3 rounded-xl
                transition-all duration-300 relative overflow-hidden group
                ${isCollapsed ? 'justify-center' : ''}
                ${isActive 
                  ? 'gradient-primary shadow-lg text-white' 
                  : theme === 'dark'
                    ? 'hover:bg-gray-700 text-gray-300 hover:text-white'
                    : 'hover:bg-primary-50 text-gray-600 hover:text-primary-dark'
                }
              `}
              title={isCollapsed ? item.label : ''}
            >
              <Icon className={`w-5 h-5 relative z-10 flex-shrink-0`} />
              {!isCollapsed && (
                <span className="font-medium relative z-10 transition-opacity duration-300">{item.label}</span>
              )}
              
              {/* Active indicator */}
              {isActive && !isCollapsed && (
                <div className="absolute right-3 w-2 h-2 bg-white rounded-full"></div>
              )}
            </button>
          );
        })}
        
        {/* Settings Button */}
        <div className={`pt-2 ${!isCollapsed ? 'border-t border-gray-700/30' : ''}`}>
          <button
            onClick={() => setCurrentPage('settings')}
            className={`
              w-full flex items-center gap-3 px-4 py-3 rounded-xl
              transition-all duration-300 relative overflow-hidden group
              ${isCollapsed ? 'justify-center' : ''}
              ${currentPage === 'settings'
                ? 'gradient-primary shadow-lg text-white'
                : theme === 'dark'
                  ? 'hover:bg-gray-700 text-gray-400 hover:text-white'
                  : 'hover:bg-warm-light/50 text-gray-500 hover:text-gray-700'
              }
            `}
            title={isCollapsed ? t('settings', language) : ''}
          >
            <Settings className={`w-5 h-5 relative z-10 flex-shrink-0`} />
            {!isCollapsed && (
              <span className="font-medium relative z-10 transition-opacity duration-300">{t('settings', language)}</span>
            )}
            {/* Active indicator */}
            {currentPage === 'settings' && !isCollapsed && (
              <div className="absolute right-3 w-2 h-2 bg-white rounded-full"></div>
            )}
          </button>
        </div>
      </nav>

      {/* Collapse Toggle Button */}
      <div className={`p-4 border-t ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'}`}>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={`
            w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs
            transition-all duration-300
            ${isCollapsed ? 'justify-center' : ''}
            ${theme === 'dark'
              ? 'bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-gray-300'
              : 'bg-primary-50/50 hover:bg-primary-100/50 text-primary-dark/70 hover:text-primary-dark'
            }
          `}
          title={isCollapsed ? t('expand', language) : t('collapse', language)}
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span className="font-medium">{t('collapse', language)}</span>
            </>
          )}
        </button>
      </div>

      {/* Footer */}
      {!isCollapsed && (
        <div className={`px-4 pb-4 border-t ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'} pt-4`}>
          <div className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'} space-y-1`}>
            <p>© 2026 Pet Monitor</p>
            <p className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-primary rounded-full"></span>
              Version 1.0.0
            </p>
          </div>
        </div>
      )}
    </aside>
  );
}