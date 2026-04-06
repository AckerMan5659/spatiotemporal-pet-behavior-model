import React, { useState } from 'react';
import { User, Mail, Phone, Building, Camera, Save, Shield, Bell, Palette, Globe, Lock, Key, Database, Activity, Info, Upload, Check } from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';

interface SettingsProps {
  language: Language;
  theme: Theme;
}

export function Settings({ language, theme }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<'profile' | 'security' | 'notifications' | 'system' | 'about'>('profile');
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);

  // User profile state
  const [profile, setProfile] = useState({
    name: 'Admin User',
    email: 'admin@petmonitor.com',
    phone: '+86 138 0000 0000',
    company: '智能宠物科技有限公司',
    role: '系统管理员',
    location: '中国·深圳'
  });

  // Security settings
  const [security, setSecurity] = useState({
    twoFactorAuth: true,
    sessionTimeout: '30',
    loginAlerts: true
  });

  // Notification settings
  const [notifications, setNotifications] = useState({
    emailAlerts: true,
    pushAlerts: true,
    abnormalBehavior: true,
    systemUpdates: true,
    weeklyReport: true,
    soundEnabled: true
  });

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatarPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSave = () => {
    setShowSuccess(true);
    setTimeout(() => setShowSuccess(false), 3000);
  };

  const tabs = [
    { id: 'profile' as const, label: '个人资料', icon: User },
    { id: 'security' as const, label: '安全设置', icon: Shield },
    { id: 'notifications' as const, label: '通知设置', icon: Bell },
    { id: 'system' as const, label: '系统偏好', icon: Palette },
    { id: 'about' as const, label: '关于系统', icon: Info }
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="mb-6">
        <h1 className={`text-2xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
          系统设置
        </h1>
        <p className={`${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          管理您的账户、偏好设置和系统配置
        </p>
      </div>

      {/* Success Toast */}
      {showSuccess && (
        <div className="fixed top-24 right-6 z-50 animate-slideInRight">
          <div className="bg-green-500 text-white px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3">
            <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
              <Check className="w-5 h-5" />
            </div>
            <div>
              <p className="font-bold">保存成功！</p>
              <p className="text-sm opacity-90">您的设置已更新</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex gap-6 overflow-hidden">
        {/* Sidebar Tabs */}
        <div className={`w-64 ${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl border p-4 space-y-2 h-fit`}>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all
                  ${isActive
                    ? 'gradient-primary text-white shadow-lg'
                    : theme === 'dark'
                      ? 'hover:bg-gray-700 text-gray-300'
                      : 'hover:bg-primary-50 text-gray-600'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto">
          <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-2xl border p-6`}>
            
            {/* Profile Tab */}
            {activeTab === 'profile' && (
              <div className="space-y-6">
                <div>
                  <h2 className={`text-xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    个人资料
                  </h2>
                  <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    管理您的个人信息和头像
                  </p>
                </div>

                {/* Avatar Upload */}
                <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-6`}>
                  <label className={`block text-sm font-medium mb-4 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                    头像
                  </label>
                  <div className="flex items-center gap-6">
                    <div className="relative">
                      {avatarPreview ? (
                        <img src={avatarPreview} alt="Avatar" className="w-24 h-24 rounded-full object-cover shadow-lg" />
                      ) : (
                        <div className="w-24 h-24 gradient-primary rounded-full flex items-center justify-center shadow-lg">
                          <User className="w-12 h-12 text-white" />
                        </div>
                      )}
                      <label className="absolute bottom-0 right-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center cursor-pointer shadow-lg hover:scale-110 transition-transform">
                        <Camera className="w-4 h-4 text-white" />
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleAvatarChange}
                          className="hidden"
                        />
                      </label>
                    </div>
                    <div className="flex-1">
                      <p className={`font-medium mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                        更改头像
                      </p>
                      <p className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                        支持 JPG, PNG 格式，文件大小不超过 2MB
                      </p>
                      <label className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer transition-all ${theme === 'dark' ? 'bg-gray-600 hover:bg-gray-500 text-white' : 'bg-white hover:bg-gray-50 text-primary-dark border border-primary'}`}>
                        <Upload className="w-4 h-4" />
                        <span className="text-sm font-medium">上传新头像</span>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleAvatarChange}
                          className="hidden"
                        />
                      </label>
                    </div>
                  </div>
                </div>

                {/* Profile Form */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      <User className="w-4 h-4 inline mr-2" />
                      姓名
                    </label>
                    <input
                      type="text"
                      value={profile.name}
                      onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                    />
                  </div>

                  <div>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      <Mail className="w-4 h-4 inline mr-2" />
                      邮箱
                    </label>
                    <input
                      type="email"
                      value={profile.email}
                      onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                    />
                  </div>

                  <div>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      <Phone className="w-4 h-4 inline mr-2" />
                      手机号码
                    </label>
                    <input
                      type="tel"
                      value={profile.phone}
                      onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                    />
                  </div>

                  <div>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      <Building className="w-4 h-4 inline mr-2" />
                      公司/组织
                    </label>
                    <input
                      type="text"
                      value={profile.company}
                      onChange={(e) => setProfile({ ...profile, company: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                    />
                  </div>

                  <div>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      角色
                    </label>
                    <input
                      type="text"
                      value={profile.role}
                      onChange={(e) => setProfile({ ...profile, role: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                    />
                  </div>

                  <div>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      <Globe className="w-4 h-4 inline mr-2" />
                      地区
                    </label>
                    <input
                      type="text"
                      value={profile.location}
                      onChange={(e) => setProfile({ ...profile, location: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Security Tab */}
            {activeTab === 'security' && (
              <div className="space-y-6">
                <div>
                  <h2 className={`text-xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    安全设置
                  </h2>
                  <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    管理您的密码和安全选项
                  </p>
                </div>

                {/* Password Change */}
                <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-warm-light/30'} rounded-xl p-6`}>
                  <h3 className={`font-bold mb-4 flex items-center gap-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    <Key className="w-5 h-5" />
                    修改密码
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                        当前密码
                      </label>
                      <input
                        type="password"
                        placeholder="输入当前密码"
                        className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                      />
                    </div>
                    <div>
                      <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                        新密码
                      </label>
                      <input
                        type="password"
                        placeholder="输入新密码"
                        className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                      />
                    </div>
                    <div>
                      <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                        确认新密码
                      </label>
                      <input
                        type="password"
                        placeholder="再次输入新密码"
                        className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white focus:border-primary' : 'bg-white border-gray-300 text-gray-800 focus:border-primary'}`}
                      />
                    </div>
                  </div>
                </div>

                {/* Security Options */}
                <div className="space-y-4">
                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4 flex items-center justify-between`}>
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
                        <Lock className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <p className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                          双因素认证
                        </p>
                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                          增强账户安全性
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => setSecurity({ ...security, twoFactorAuth: !security.twoFactorAuth })}
                      className={`w-12 h-6 rounded-full transition-all ${security.twoFactorAuth ? 'bg-primary' : theme === 'dark' ? 'bg-gray-600' : 'bg-gray-300'}`}
                    >
                      <div className={`w-5 h-5 bg-white rounded-full transition-all ${security.twoFactorAuth ? 'ml-6' : 'ml-1'}`} />
                    </button>
                  </div>

                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4 flex items-center justify-between`}>
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-secondary rounded-lg flex items-center justify-center">
                        <Activity className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <p className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                          登录提醒
                        </p>
                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                          新设备登录时通知您
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => setSecurity({ ...security, loginAlerts: !security.loginAlerts })}
                      className={`w-12 h-6 rounded-full transition-all ${security.loginAlerts ? 'bg-primary' : theme === 'dark' ? 'bg-gray-600' : 'bg-gray-300'}`}
                    >
                      <div className={`w-5 h-5 bg-white rounded-full transition-all ${security.loginAlerts ? 'ml-6' : 'ml-1'}`} />
                    </button>
                  </div>

                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4`}>
                    <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`}>
                      会话超时时间（分钟）
                    </label>
                    <select
                      value={security.sessionTimeout}
                      onChange={(e) => setSecurity({ ...security, sessionTimeout: e.target.value })}
                      className={`w-full px-4 py-3 rounded-xl border transition-all ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-800'}`}
                    >
                      <option value="15">15 分钟</option>
                      <option value="30">30 分钟</option>
                      <option value="60">60 分钟</option>
                      <option value="120">120 分钟</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* Notifications Tab */}
            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <div>
                  <h2 className={`text-xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    通知设置
                  </h2>
                  <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    管理您接收通知的方式和类型
                  </p>
                </div>

                <div className="space-y-4">
                  {[
                    { key: 'emailAlerts', label: '邮件通知', desc: '通过邮件接收重要更新' },
                    { key: 'pushAlerts', label: '推送通知', desc: '浏览器桌面推送提醒' },
                    { key: 'abnormalBehavior', label: '异常行为告警', desc: '检测到异常行为时通知' },
                    { key: 'systemUpdates', label: '系统更新通知', desc: '新功能和系统维护通知' },
                    { key: 'weeklyReport', label: '每周数据报告', desc: '每周发送统计报告' },
                    { key: 'soundEnabled', label: '声音提示', desc: '收到通知时播放声音' }
                  ].map((item) => (
                    <div
                      key={item.key}
                      className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4 flex items-center justify-between`}
                    >
                      <div>
                        <p className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                          {item.label}
                        </p>
                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                          {item.desc}
                        </p>
                      </div>
                      <button
                        onClick={() => setNotifications({ ...notifications, [item.key]: !notifications[item.key as keyof typeof notifications] })}
                        className={`w-12 h-6 rounded-full transition-all ${notifications[item.key as keyof typeof notifications] ? 'bg-primary' : theme === 'dark' ? 'bg-gray-600' : 'bg-gray-300'}`}
                      >
                        <div className={`w-5 h-5 bg-white rounded-full transition-all ${notifications[item.key as keyof typeof notifications] ? 'ml-6' : 'ml-1'}`} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* System Preferences Tab */}
            {activeTab === 'system' && (
              <div className="space-y-6">
                <div>
                  <h2 className={`text-xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    系统偏好
                  </h2>
                  <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    自定义系统外观和行为
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-6`}>
                    <Palette className="w-8 h-8 text-primary mb-3" />
                    <h3 className={`font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                      主题模式
                    </h3>
                    <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                      当前: {theme === 'dark' ? '黑暗模式' : '白天模式'}
                    </p>
                    <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>
                      可在顶部工具栏切换
                    </p>
                  </div>

                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-6`}>
                    <Globe className="w-8 h-8 text-primary mb-3" />
                    <h3 className={`font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                      语言
                    </h3>
                    <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                      当前: {language === 'zh-CN' ? '简体中文' : language === 'zh-HK' ? '繁體中文' : 'English'}
                    </p>
                    <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>
                      可在顶部工具栏切换
                    </p>
                  </div>

                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-6`}>
                    <Database className="w-8 h-8 text-primary mb-3" />
                    <h3 className={`font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                      数据保留
                    </h3>
                    <select className={`w-full mt-2 px-3 py-2 rounded-lg border ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'}`}>
                      <option>30 天</option>
                      <option>60 天</option>
                      <option>90 天</option>
                      <option>永久保存</option>
                    </select>
                  </div>

                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-6`}>
                    <Activity className="w-8 h-8 text-primary mb-3" />
                    <h3 className={`font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                      性能模式
                    </h3>
                    <select className={`w-full mt-2 px-3 py-2 rounded-lg border ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300'}`}>
                      <option>高性能</option>
                      <option>平衡</option>
                      <option>省电</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* About Tab */}
            {activeTab === 'about' && (
              <div className="space-y-6">
                <div>
                  <h2 className={`text-xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    关于系统
                  </h2>
                  <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    系统信息和版本详情
                  </p>
                </div>

                <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-gradient-to-br from-primary-50 to-secondary-light/50'} rounded-xl p-8 text-center`}>
                  <div className="w-20 h-20 gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-xl">
                    <Activity className="w-10 h-10 text-white" />
                  </div>
                  <h3 className={`text-2xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    智能宠物监控系统
                  </h3>
                  <p className={`text-sm mb-6 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                    Intelligent Pet Monitoring & Behavior Analysis Platform
                  </p>
                  <div className="inline-flex items-center gap-2 bg-primary text-white px-4 py-2 rounded-full font-medium shadow-lg">
                    Version 1.0.0
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4`}>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>发布日期</p>
                    <p className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>2026-02-02</p>
                  </div>
                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4`}>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>技术栈</p>
                    <p className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>React + Tailwind CSS</p>
                  </div>
                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4`}>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>许可证</p>
                    <p className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>MIT License</p>
                  </div>
                  <div className={`${theme === 'dark' ? 'bg-gray-700/50' : 'bg-primary-50'} rounded-xl p-4`}>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>支持</p>
                    <p className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>support@petmonitor.com</p>
                  </div>
                </div>

                <div className={`${theme === 'dark' ? 'bg-gray-700/50 border-gray-600' : 'bg-white border-gray-200'} rounded-xl p-6 border`}>
                  <h4 className={`font-bold mb-3 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                    更新日志
                  </h4>
                  <div className="space-y-3">
                    <div className="flex gap-3">
                      <div className="w-2 h-2 bg-primary rounded-full mt-1.5 flex-shrink-0" />
                      <div>
                        <p className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
                          v1.0.0 - 初始版本
                        </p>
                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                          完整的仪表盘、数据管理和监控功能
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Save Button */}
            {(activeTab === 'profile' || activeTab === 'security' || activeTab === 'notifications') && (
              <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-700">
                <button className={`px-6 py-3 rounded-xl font-medium transition-all ${theme === 'dark' ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}>
                  取消
                </button>
                <button
                  onClick={handleSave}
                  className="px-6 py-3 gradient-primary text-white rounded-xl font-medium shadow-lg hover:shadow-xl transition-all flex items-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  保存更改
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slideInRight {
          from {
            opacity: 0;
            transform: translateX(100px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        .animate-slideInRight {
          animation: slideInRight 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}
