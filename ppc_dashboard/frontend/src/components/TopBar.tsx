import React, { useState } from 'react';
import { Type, Globe, User, ChevronDown, Sun, Moon, Bell } from 'lucide-react';
import { Language, FontSize, Theme } from '../App';
import { t } from '../utils/translations';

interface TopBarProps {
  language: Language;
  setLanguage: (lang: Language) => void;
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export function TopBar({ language, setLanguage, fontSize, setFontSize, theme, setTheme }: TopBarProps) {
  const [showLangMenu, setShowLangMenu] = useState(false);
  const [showFontMenu, setShowFontMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const languages: { code: Language; label: string }[] = [
    { code: 'zh-CN', label: '简体中文' },
    { code: 'zh-HK', label: '繁體中文 (香港)' },
    { code: 'en', label: 'English' }
  ];

  const fontSizes: { code: FontSize; label: string }[] = [
    { code: 'small', label: t('fontSizeSmall', language) },
    { code: 'medium', label: t('fontSizeMedium', language) },
    { code: 'large', label: t('fontSizeLarge', language) }
  ];

  const notifications = [
    { id: 1, title: 'CAM-03 异常行为检测', message: '检测到宠物异常行为，置信度 88.7%', time: '2分钟前', unread: true },
    { id: 2, title: 'CAM-01 进食行为', message: '宠物开始进食，时长 15 分钟', time: '15分钟前', unread: true },
    { id: 3, title: '系统更新通知', message: '行为识别模型已更新至 v2.3', time: '1小时前', unread: true },
    { id: 4, title: 'CAM-05 离线警告', message: '摄像头连接中断，请检查网络', time: '2小时前', unread: true },
    { id: 5, title: '每日报告生成', message: '昨日宠物行为分析报告已生成', time: '3小时前', unread: true },
  ];

  const unreadCount = notifications.filter(n => n.unread).length;

  return (
    <header className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-6 py-4 flex items-center justify-between shadow-sm`}>
      <div>
        <h1 className={`text-xl font-bold ${theme === 'dark' ? 'text-white' : 'text-primary-dark'}`}>
          {t('systemName', language)}
        </h1>
      </div>

      <div className="flex items-center gap-3">
        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className={`relative flex items-center gap-2 rounded-lg px-4 py-2 transition-all text-white shadow-sm hover:shadow-md ${theme === 'dark' ? 'bg-primary-700 hover:bg-primary-800' : 'bg-primary hover:bg-primary-dark'}`}
            title={t('notifications', language)}
          >
            <Bell className="w-4 h-4" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center shadow-md">
                {unreadCount}
              </span>
            )}
          </button>

          {showNotifications && (
            <>
              <div 
                className="fixed inset-0 z-10" 
                onClick={() => setShowNotifications(false)}
              />
              <div className={`absolute right-0 mt-2 w-80 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl shadow-xl border py-3 z-20 max-h-96 overflow-y-auto`}>
                <div className="px-4 pb-3 border-b border-gray-700 mb-2">
                  <h3 className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    {t('notifications', language)}
                  </h3>
                  <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'} mt-1`}>
                    {unreadCount} 条未读消息
                  </p>
                </div>
                <div className="space-y-1">
                  {notifications.map((notif) => (
                    <div
                      key={notif.id}
                      className={`
                        px-4 py-3 transition-all cursor-pointer relative
                        ${notif.unread 
                          ? theme === 'dark' 
                            ? 'bg-primary-900/20 hover:bg-primary-900/30' 
                            : 'bg-primary-50 hover:bg-primary-100'
                          : theme === 'dark'
                            ? 'hover:bg-gray-700/50'
                            : 'hover:bg-gray-50'
                        }
                      `}
                    >
                      {notif.unread && (
                        <div className="absolute left-2 top-1/2 -translate-y-1/2 w-2 h-2 bg-primary rounded-full"></div>
                      )}
                      <div className={`${notif.unread ? 'ml-3' : ''}`}>
                        <p className={`text-sm font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                          {notif.title}
                        </p>
                        <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mt-1`}>
                          {notif.message}
                        </p>
                        <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'} mt-1`}>
                          {notif.time}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className={`px-4 pt-3 border-t ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'} mt-2`}>
                  <button className={`text-xs font-medium w-full py-2 rounded-lg transition-all ${theme === 'dark' ? 'text-primary hover:bg-gray-700' : 'text-primary-dark hover:bg-primary-50'}`}>
                    查看全部消息
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Font Size Control */}
        <div className="relative">
          <button
            onClick={() => setShowFontMenu(!showFontMenu)}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-all text-white shadow-sm hover:shadow-md ${theme === 'dark' ? 'bg-primary-700 hover:bg-primary-800' : 'bg-primary hover:bg-primary-dark'}`}
            title={t('fontSize', language)}
          >
            <Type className="w-4 h-4" />
            <span className="text-sm font-medium">
              {fontSizes.find(f => f.code === fontSize)?.label}
            </span>
            <ChevronDown className="w-4 h-4 text-white/70" />
          </button>

          {showFontMenu && (
            <>
              <div 
                className="fixed inset-0 z-10" 
                onClick={() => setShowFontMenu(false)}
              />
              <div className={`absolute right-0 mt-2 w-36 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl shadow-xl border py-2 z-20`}>
                {fontSizes.map((size) => (
                  <button
                    key={size.code}
                    onClick={() => {
                      setFontSize(size.code);
                      setShowFontMenu(false);
                    }}
                    className={`
                      w-full text-left px-4 py-2 text-sm transition-all
                      ${fontSize === size.code 
                        ? 'gradient-primary text-white font-medium' 
                        : theme === 'dark'
                          ? 'text-gray-300 hover:bg-gray-700'
                          : 'text-gray-700 hover:bg-primary-50'
                      }
                    `}
                  >
                    {size.label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Theme Toggle */}
        <button
          onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-all shadow-sm hover:shadow-md text-white ${theme === 'dark' ? 'bg-primary-700 hover:bg-primary-800' : 'bg-primary hover:bg-primary-dark'}`}
          title={theme === 'light' ? t('darkMode', language) : t('lightMode', language)}
        >
          {theme === 'light' ? (
            <>
              <Moon className="w-4 h-4" />
              <span className="text-sm font-medium">{t('darkMode', language)}</span>
            </>
          ) : (
            <>
              <Sun className="w-4 h-4" />
              <span className="text-sm font-medium">{t('lightMode', language)}</span>
            </>
          )}
        </button>

        {/* Language Selector */}
        <div className="relative">
          <button
            onClick={() => setShowLangMenu(!showLangMenu)}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-all text-white shadow-sm hover:shadow-md ${theme === 'dark' ? 'bg-primary-700 hover:bg-primary-800' : 'bg-primary hover:bg-primary-dark'}`}
          >
            <Globe className="w-4 h-4" />
            <span className="text-sm font-medium">
              {languages.find(l => l.code === language)?.label}
            </span>
            <ChevronDown className="w-4 h-4 text-white/70" />
          </button>

          {showLangMenu && (
            <>
              <div 
                className="fixed inset-0 z-10" 
                onClick={() => setShowLangMenu(false)}
              />
              <div className={`absolute right-0 mt-2 w-48 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl shadow-xl border py-2 z-20`}>
                {languages.map((lang) => (
                  <button
                    key={lang.code}
                    onClick={() => {
                      setLanguage(lang.code);
                      setShowLangMenu(false);
                    }}
                    className={`
                      w-full text-left px-4 py-2 text-sm transition-all
                      ${language === lang.code 
                        ? 'gradient-primary text-white font-medium' 
                        : theme === 'dark'
                          ? 'text-gray-300 hover:bg-gray-700'
                          : 'text-gray-700 hover:bg-primary-50'
                      }
                    `}
                  >
                    {lang.label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        {/* User Menu */}
        <div className={`flex items-center gap-3 pl-3 ml-3 border-l ${theme === 'dark' ? 'border-gray-700' : 'border-primary/20'}`}>
          <div className="text-right">
            <p className={`text-sm font-medium ${theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>Admin User</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>管理员</p>
          </div>
          <div className="w-10 h-10 gradient-primary rounded-full flex items-center justify-center shadow-lg relative overflow-hidden group cursor-pointer">
            <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent group-hover:from-white/30 transition-all"></div>
            <User className="w-5 h-5 text-white relative z-10" />
          </div>
        </div>
      </div>
    </header>
  );
}