import React, { useState } from 'react';
import { Search, Grid, List, Plus, ArrowLeft, Calendar, Weight, Ruler, Heart, Activity, Image as ImageIcon, Video as VideoIcon, Edit, Trash2, X, Upload, Check } from 'lucide-react';
import { Language, Theme } from '../App';
import { t } from '../utils/translations';

interface PetDataManagementProps {
  language: Language;
  theme: Theme;
}

interface PetVideo {
  id: string;
  name: string;
  url: string;
  thumbnail?: string;
  duration?: string;
}

interface Pet {
  id: number;
  name: string;
  breed: string;
  species: 'dog' | 'cat';
  age: number;
  weight: number;
  gender: 'male' | 'female';
  color: string;
  imageUrl: string;
  birthDate: string;
  microchipId: string;
  healthStatus: 'healthy' | 'monitoring' | 'sick';
  lastCheckup: string;
  vaccinated: boolean;
  photos: string[];
  videos: PetVideo[];
  notes: string;
  adoptionDate: string;
}

export function PetDataManagement({ language, theme }: PetDataManagementProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPet, setSelectedPet] = useState<Pet | null>(null);
  const [showAddPetModal, setShowAddPetModal] = useState(false);
  const [pets, setPets] = useState<Pet[]>(initialPets);

  // Mock pet data with real video URLs
  const filteredPets = pets.filter(pet => 
    pet.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    pet.breed.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handlePetClick = (pet: Pet) => {
    setSelectedPet(pet);
  };

  const handleBackToList = () => {
    setSelectedPet(null);
  };

  const handleAddPet = (newPet: Omit<Pet, 'id'>) => {
    const pet: Pet = {
      ...newPet,
      id: Math.max(...pets.map(p => p.id)) + 1
    };
    setPets([...pets, pet]);
    setShowAddPetModal(false);
  };

  if (selectedPet) {
    return <PetDetail pet={selectedPet} onBack={handleBackToList} language={language} theme={theme} />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
          {t('petDataManagement', language)}
        </h2>
        <button 
          onClick={() => setShowAddPetModal(true)}
          className="gradient-primary text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Plus className="w-5 h-5" />
          {t('addPet', language)}
        </button>
      </div>

      {/* Search and View Toggle */}
      <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl shadow-sm border p-4`}>
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 relative">
            <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
            <input
              type="text"
              placeholder={t('searchPets', language)}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all
                ${theme === 'dark' 
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                  : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                }`}
            />
          </div>
          
          <div className={`flex items-center gap-1 p-1 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded transition-all ${
                viewMode === 'grid'
                  ? 'gradient-primary text-white shadow-lg'
                  : theme === 'dark'
                    ? 'text-gray-400 hover:text-white'
                    : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Grid className="w-5 h-5" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded transition-all ${
                viewMode === 'list'
                  ? 'gradient-primary text-white shadow-lg'
                  : theme === 'dark'
                    ? 'text-gray-400 hover:text-white'
                    : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <List className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Pet Display */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPets.map((pet) => (
            <PetCard key={pet.id} pet={pet} onClick={() => handlePetClick(pet)} theme={theme} language={language} />
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredPets.map((pet) => (
            <PetListItem key={pet.id} pet={pet} onClick={() => handlePetClick(pet)} theme={theme} language={language} />
          ))}
        </div>
      )}

      {/* Add Pet Modal */}
      {showAddPetModal && (
        <AddPetModal
          onClose={() => setShowAddPetModal(false)}
          onSave={handleAddPet}
          theme={theme}
          language={language}
        />
      )}
    </div>
  );
}

interface PetCardProps {
  pet: Pet;
  onClick: () => void;
  theme: Theme;
  language: Language;
}

function PetCard({ pet, onClick, theme, language }: PetCardProps) {
  const healthStatusColors = {
    healthy: theme === 'dark' ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700',
    monitoring: theme === 'dark' ? 'bg-yellow-900/50 text-yellow-300' : 'bg-yellow-100 text-yellow-700',
    sick: theme === 'dark' ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700'
  };

  return (
    <div
      onClick={onClick}
      className={`
        ${theme === 'dark' ? 'bg-gray-800 border-gray-700 hover:border-primary' : 'bg-white border-gray-200 hover:border-primary'}
        rounded-xl border shadow-sm hover:shadow-xl transition-all duration-300 cursor-pointer overflow-hidden group
      `}
    >
      <div className="relative h-48 overflow-hidden">
        <img 
          src={pet.imageUrl} 
          alt={pet.name}
          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
        />
        <div className="absolute top-3 right-3">
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${healthStatusColors[pet.healthStatus]}`}>
            {t(pet.healthStatus, language)}
          </span>
        </div>
      </div>
      
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className={`text-xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              {pet.name}
            </h3>
            <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              {pet.breed}
            </p>
          </div>
          <span className={`text-2xl`}>
            {pet.species === 'dog' ? '🐕' : '🐱'}
          </span>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Calendar className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
            <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
              {pet.age} {t('yearsOld', language)} • {pet.gender === 'male' ? '♂' : '♀'}
            </span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Weight className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
            <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
              {pet.weight} kg
            </span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Heart className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
            <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
              {pet.vaccinated ? t('vaccinated', language) : t('notVaccinated', language)}
            </span>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 text-sm">
              <ImageIcon className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
              <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>{pet.photos.length}</span>
            </div>
            <div className="flex items-center gap-1 text-sm">
              <VideoIcon className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
              <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>{pet.videos.length}</span>
            </div>
          </div>
          <button className="text-primary hover:text-primary-dark text-sm font-medium">
            {t('viewDetails', language)} →
          </button>
        </div>
      </div>
    </div>
  );
}

interface PetListItemProps {
  pet: Pet;
  onClick: () => void;
  theme: Theme;
  language: Language;
}

function PetListItem({ pet, onClick, theme, language }: PetListItemProps) {
  const healthStatusColors = {
    healthy: theme === 'dark' ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700',
    monitoring: theme === 'dark' ? 'bg-yellow-900/50 text-yellow-300' : 'bg-yellow-100 text-yellow-700',
    sick: theme === 'dark' ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700'
  };

  return (
    <div
      onClick={onClick}
      className={`
        ${theme === 'dark' ? 'bg-gray-800 border-gray-700 hover:border-primary' : 'bg-white border-gray-200 hover:border-primary'}
        rounded-xl border shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer p-4
      `}
    >
      <div className="flex items-center gap-4">
        <div className="w-24 h-24 rounded-lg overflow-hidden flex-shrink-0">
          <img 
            src={pet.imageUrl} 
            alt={pet.name}
            className="w-full h-full object-cover"
          />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-3">
              <h3 className={`text-xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                {pet.name}
              </h3>
              <span className="text-2xl">
                {pet.species === 'dog' ? '🐕' : '🐱'}
              </span>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${healthStatusColors[pet.healthStatus]}`}>
              {t(pet.healthStatus, language)}
            </span>
          </div>
          
          <p className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            {pet.breed} • {pet.color}
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2 text-sm">
              <Calendar className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
              <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                {pet.age} {t('yearsOld', language)}
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Weight className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
              <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                {pet.weight} kg
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <ImageIcon className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
              <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                {pet.photos.length} {t('photos', language)}
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <VideoIcon className={`w-4 h-4 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`} />
              <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                {pet.videos.length} {t('videos', language)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface PetDetailProps {
  pet: Pet;
  onBack: () => void;
  language: Language;
  theme: Theme;
}

function PetDetail({ pet, onBack, language, theme }: PetDetailProps) {
  const [activeTab, setActiveTab] = useState<'info' | 'photos' | 'videos' | 'health'>('info');
  const [playingVideo, setPlayingVideo] = useState<PetVideo | null>(null);

  const healthStatusColors = {
    healthy: theme === 'dark' ? 'bg-green-900/50 text-green-300 border-green-700' : 'bg-green-100 text-green-700 border-green-300',
    monitoring: theme === 'dark' ? 'bg-yellow-900/50 text-yellow-300 border-yellow-700' : 'bg-yellow-100 text-yellow-700 border-yellow-300',
    sick: theme === 'dark' ? 'bg-red-900/50 text-red-300 border-red-700' : 'bg-red-100 text-red-700 border-red-300'
  };

  return (
    <div className="space-y-6">
      {/* Header with Back Button */}
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className={`p-2 rounded-lg transition-colors ${
            theme === 'dark' 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
        >
          <ArrowLeft className="w-6 h-6" />
        </button>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h2 className={`text-3xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
              {pet.name}
            </h2>
            <span className="text-3xl">
              {pet.species === 'dog' ? '🐕' : '🐱'}
            </span>
          </div>
          <p className={`text-lg mt-1 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            {pet.breed}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className={`p-2 rounded-lg transition-colors ${
            theme === 'dark' ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
          }`}>
            <Edit className="w-5 h-5" />
          </button>
          <button className={`p-2 rounded-lg transition-colors ${
            theme === 'dark' ? 'hover:bg-red-900/50 text-red-400' : 'hover:bg-red-50 text-red-600'
          }`}>
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Photo */}
        <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border shadow-sm overflow-hidden`}>
          <div className="relative h-80">
            <img 
              src={pet.imageUrl} 
              alt={pet.name}
              className="w-full h-full object-cover"
            />
            <div className="absolute top-4 right-4">
              <span className={`px-4 py-2 rounded-full text-sm font-medium border ${healthStatusColors[pet.healthStatus]}`}>
                {t(pet.healthStatus, language)}
              </span>
            </div>
          </div>
          
          <div className="p-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>{t('age', language)}</p>
                <p className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  {pet.age} {t('yearsOld', language)}
                </p>
              </div>
              <div>
                <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>{t('gender', language)}</p>
                <p className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  {pet.gender === 'male' ? t('male', language) : t('female', language)}
                </p>
              </div>
              <div>
                <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>{t('weight', language)}</p>
                <p className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  {pet.weight} kg
                </p>
              </div>
              <div>
                <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>{t('color', language)}</p>
                <p className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  {pet.color}
                </p>
              </div>
            </div>

            <div className="pt-4 border-t">
              <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'} mb-1`}>{t('microchipId', language)}</p>
              <p className={`font-mono text-sm ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                {pet.microchipId}
              </p>
            </div>

            <div>
              <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'} mb-1`}>{t('adoptionDate', language)}</p>
              <p className={`text-sm ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                {pet.adoptionDate}
              </p>
            </div>
          </div>
        </div>

        {/* Right Column - Details */}
        <div className="lg:col-span-2 space-y-6">
          {/* Tabs */}
          <div className={`${theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border shadow-sm`}>
            <div className="flex border-b">
              <button
                onClick={() => setActiveTab('info')}
                className={`flex-1 px-6 py-3 font-medium transition-colors ${
                  activeTab === 'info'
                    ? 'text-primary border-b-2 border-primary'
                    : theme === 'dark'
                      ? 'text-gray-400 hover:text-gray-300'
                      : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {t('basicInfo', language)}
              </button>
              <button
                onClick={() => setActiveTab('photos')}
                className={`flex-1 px-6 py-3 font-medium transition-colors ${
                  activeTab === 'photos'
                    ? 'text-primary border-b-2 border-primary'
                    : theme === 'dark'
                      ? 'text-gray-400 hover:text-gray-300'
                      : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {t('photos', language)} ({pet.photos.length})
              </button>
              <button
                onClick={() => setActiveTab('videos')}
                className={`flex-1 px-6 py-3 font-medium transition-colors ${
                  activeTab === 'videos'
                    ? 'text-primary border-b-2 border-primary'
                    : theme === 'dark'
                      ? 'text-gray-400 hover:text-gray-300'
                      : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {t('videos', language)} ({pet.videos.length})
              </button>
              <button
                onClick={() => setActiveTab('health')}
                className={`flex-1 px-6 py-3 font-medium transition-colors ${
                  activeTab === 'health'
                    ? 'text-primary border-b-2 border-primary'
                    : theme === 'dark'
                      ? 'text-gray-400 hover:text-gray-300'
                      : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {t('healthRecords', language)}
              </button>
            </div>

            <div className="p-6">
              {activeTab === 'info' && (
                <div className="space-y-6">
                  <div>
                    <h3 className={`text-lg font-semibold mb-3 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      {t('notes', language)}
                    </h3>
                    <p className={`leading-relaxed ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                      {pet.notes}
                    </p>
                  </div>

                  <div>
                    <h3 className={`text-lg font-semibold mb-3 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      {t('birthdayInfo', language)}
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-50'}`}>
                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'} mb-1`}>
                          {t('birthDate', language)}
                        </p>
                        <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                          {pet.birthDate}
                        </p>
                      </div>
                      <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-50'}`}>
                        <p className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'} mb-1`}>
                          {t('age', language)}
                        </p>
                        <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                          {pet.age} {t('yearsOld', language)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'photos' && (
                <div className="grid grid-cols-2 gap-4">
                  {pet.photos.map((photo, index) => (
                    <div key={index} className="aspect-square rounded-lg overflow-hidden group cursor-pointer">
                      <img 
                        src={photo} 
                        alt={`${pet.name} photo ${index + 1}`}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                      />
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'videos' && (
                <div className="space-y-4">
                  {pet.videos.map((video) => (
                    <div key={video.id} className={`p-4 rounded-lg border ${theme === 'dark' ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'}`}>
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                          <VideoIcon className="w-6 h-6 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                            {video.name}
                          </p>
                          <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                            {video.duration || 'Duration: --:--'}
                          </p>
                        </div>
                        <button 
                          onClick={() => setPlayingVideo(video)}
                          className="gradient-primary text-white px-4 py-2 rounded-lg text-sm hover:opacity-90 transition-opacity"
                        >
                          {t('play', language)}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'health' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className={`p-4 rounded-lg border ${healthStatusColors[pet.healthStatus]}`}>
                      <div className="flex items-center gap-3 mb-2">
                        <Heart className="w-5 h-5" />
                        <p className="font-semibold">{t('healthStatus', language)}</p>
                      </div>
                      <p className="text-lg font-bold">
                        {t(pet.healthStatus, language)}
                      </p>
                    </div>
                    <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-50'}`}>
                      <div className="flex items-center gap-3 mb-2">
                        <Activity className="w-5 h-5" />
                        <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                          {t('vaccination', language)}
                        </p>
                      </div>
                      <p className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                        {pet.vaccinated ? t('completed', language) : t('incomplete', language)}
                      </p>
                    </div>
                  </div>

                  <div className={`p-4 rounded-lg ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-50'}`}>
                    <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} mb-1`}>
                      {t('lastCheckup', language)}
                    </p>
                    <p className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      {pet.lastCheckup}
                    </p>
                  </div>

                  <div>
                    <h3 className={`text-lg font-semibold mb-3 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      {t('healthNotes', language)}
                    </h3>
                    <p className={`leading-relaxed ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                      {pet.notes}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Video Player Modal */}
      {playingVideo && (
        <VideoPlayerModal
          video={playingVideo}
          onClose={() => setPlayingVideo(null)}
          theme={theme}
          language={language}
        />
      )}
    </div>
  );
}

interface VideoPlayerModalProps {
  video: PetVideo;
  onClose: () => void;
  theme: Theme;
  language: Language;
}

function VideoPlayerModal({ video, onClose, theme, language }: VideoPlayerModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80">
      <div className={`relative w-full max-w-4xl ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'} rounded-xl overflow-hidden shadow-2xl`}>
        <div className={`flex items-center justify-between p-4 border-b ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'}`}>
          <h3 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
            {video.name}
          </h3>
          <button
            onClick={onClose}
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
            }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="aspect-video bg-black">
          <video 
            src={video.url} 
            controls 
            autoPlay
            className="w-full h-full"
          >
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    </div>
  );
}

interface AddPetModalProps {
  onClose: () => void;
  onSave: (pet: Omit<Pet, 'id'>) => void;
  theme: Theme;
  language: Language;
}

function AddPetModal({ onClose, onSave, theme, language }: AddPetModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    breed: '',
    species: 'dog' as 'dog' | 'cat',
    age: 0,
    weight: 0,
    gender: 'male' as 'male' | 'female',
    color: '',
    birthDate: '',
    microchipId: '',
    healthStatus: 'healthy' as 'healthy' | 'monitoring' | 'sick',
    lastCheckup: '',
    vaccinated: false,
    notes: '',
    adoptionDate: ''
  });

  const [imageUrl, setImageUrl] = useState('');
  const [photos, setPhotos] = useState<string[]>([]);
  const [videoUrl, setVideoUrl] = useState('');
  const [videoName, setVideoName] = useState('');
  const [videos, setVideos] = useState<PetVideo[]>([]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const newPet: Omit<Pet, 'id'> = {
      ...formData,
      imageUrl: imageUrl || 'https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a?w=800',
      photos: photos.length > 0 ? photos : [imageUrl || 'https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a?w=800'],
      videos: videos
    };

    onSave(newPet);
  };

  const addPhoto = () => {
    if (imageUrl && !photos.includes(imageUrl)) {
      setPhotos([...photos, imageUrl]);
      setImageUrl('');
    }
  };

  const removePhoto = (index: number) => {
    setPhotos(photos.filter((_, i) => i !== index));
  };

  const addVideo = () => {
    if (videoUrl && videoName) {
      const newVideo: PetVideo = {
        id: `video-${Date.now()}`,
        name: videoName,
        url: videoUrl,
        duration: '0:00'
      };
      setVideos([...videos, newVideo]);
      setVideoUrl('');
      setVideoName('');
    }
  };

  const removeVideo = (id: string) => {
    setVideos(videos.filter(v => v.id !== id));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 overflow-y-auto">
      <div className={`relative w-full max-w-4xl ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'} rounded-xl shadow-2xl my-8`}>
        <div className={`flex items-center justify-between p-6 border-b ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'}`}>
          <h2 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
            {t('addPet', language)}
          </h2>
          <button
            onClick={onClose}
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
            }`}
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6 max-h-[calc(100vh-200px)] overflow-y-auto">
          {/* Basic Information */}
          <div>
            <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              {t('basicInfo', language)}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('petName', language)} *
                </label>
                <input
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) => setFormData({...formData, name: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder={language === 'zh-CN' ? '例如：小金' : language === 'zh-HK' ? '例如：小金' : 'e.g., Max'}
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('breed', language)} *
                </label>
                <input
                  type="text"
                  required
                  value={formData.breed}
                  onChange={(e) => setFormData({...formData, breed: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder={language === 'zh-CN' ? '例如：金毛寻回犬' : language === 'zh-HK' ? '例如：金毛尋回犬' : 'e.g., Golden Retriever'}
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('species', language)} *
                </label>
                <select
                  value={formData.species}
                  onChange={(e) => setFormData({...formData, species: e.target.value as 'dog' | 'cat'})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                >
                  <option value="dog">{t('dog', language)}</option>
                  <option value="cat">{t('cat', language)}</option>
                </select>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('gender', language)} *
                </label>
                <select
                  value={formData.gender}
                  onChange={(e) => setFormData({...formData, gender: e.target.value as 'male' | 'female'})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                >
                  <option value="male">{t('male', language)}</option>
                  <option value="female">{t('female', language)}</option>
                </select>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('age', language)} *
                </label>
                <input
                  type="number"
                  required
                  min="0"
                  value={formData.age}
                  onChange={(e) => setFormData({...formData, age: parseInt(e.target.value) || 0})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder="0"
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('weight', language)} (kg) *
                </label>
                <input
                  type="number"
                  required
                  min="0"
                  step="0.1"
                  value={formData.weight}
                  onChange={(e) => setFormData({...formData, weight: parseFloat(e.target.value) || 0})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder="0.0"
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('color', language)}
                </label>
                <input
                  type="text"
                  value={formData.color}
                  onChange={(e) => setFormData({...formData, color: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder={language === 'zh-CN' ? '例如：金色' : language === 'zh-HK' ? '例如：金色' : 'e.g., Golden'}
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('birthDate', language)}
                </label>
                <input
                  type="date"
                  value={formData.birthDate}
                  onChange={(e) => setFormData({...formData, birthDate: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('microchipId', language)}
                </label>
                <input
                  type="text"
                  value={formData.microchipId}
                  onChange={(e) => setFormData({...formData, microchipId: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder="MC-XXXX-XXX-XX"
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('adoptionDate', language)}
                </label>
                <input
                  type="date"
                  value={formData.adoptionDate}
                  onChange={(e) => setFormData({...formData, adoptionDate: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                />
              </div>
            </div>
          </div>

          {/* Health Information */}
          <div>
            <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              {t('healthRecords', language)}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('healthStatus', language)}
                </label>
                <select
                  value={formData.healthStatus}
                  onChange={(e) => setFormData({...formData, healthStatus: e.target.value as any})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                >
                  <option value="healthy">{t('healthy', language)}</option>
                  <option value="monitoring">{t('monitoring', language)}</option>
                  <option value="sick">{t('sick', language)}</option>
                </select>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                  {t('lastCheckup', language)}
                </label>
                <input
                  type="date"
                  value={formData.lastCheckup}
                  onChange={(e) => setFormData({...formData, lastCheckup: e.target.value})}
                  className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                />
              </div>

              <div className="md:col-span-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.vaccinated}
                    onChange={(e) => setFormData({...formData, vaccinated: e.target.checked})}
                    className="w-5 h-5 rounded border-gray-300 text-primary focus:ring-primary"
                  />
                  <span className={`text-sm font-medium ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                    {t('vaccinated', language)}
                  </span>
                </label>
              </div>
            </div>
          </div>

          {/* Photos */}
          <div>
            <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              {t('photos', language)}
            </h3>
            <div className="space-y-3">
              <div className="flex gap-2">
                <input
                  type="url"
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  className={`flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder={language === 'zh-CN' ? '输入图片URL...' : language === 'zh-HK' ? '輸入圖片URL...' : 'Enter image URL...'}
                />
                <button
                  type="button"
                  onClick={addPhoto}
                  className="gradient-primary text-white px-4 py-2 rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  {t('add', language)}
                </button>
              </div>

              {photos.length > 0 && (
                <div className="grid grid-cols-3 md:grid-cols-4 gap-3">
                  {photos.map((photo, index) => (
                    <div key={index} className="relative group aspect-square rounded-lg overflow-hidden">
                      <img src={photo} alt={`Photo ${index + 1}`} className="w-full h-full object-cover" />
                      <button
                        type="button"
                        onClick={() => removePhoto(index)}
                        className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Videos */}
          <div>
            <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              {t('videos', language)}
            </h3>
            <div className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <input
                  type="text"
                  value={videoName}
                  onChange={(e) => setVideoName(e.target.value)}
                  className={`px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                    ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                  placeholder={language === 'zh-CN' ? '视频名称...' : language === 'zh-HK' ? '影片名稱...' : 'Video name...'}
                />
                <div className="flex gap-2">
                  <input
                    type="url"
                    value={videoUrl}
                    onChange={(e) => setVideoUrl(e.target.value)}
                    className={`flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none
                      ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
                    placeholder={language === 'zh-CN' ? '视频URL...' : language === 'zh-HK' ? '影片URL...' : 'Video URL...'}
                  />
                  <button
                    type="button"
                    onClick={addVideo}
                    className="gradient-primary text-white px-4 py-2 rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
                  >
                    <Upload className="w-4 h-4" />
                    {t('add', language)}
                  </button>
                </div>
              </div>

              {videos.length > 0 && (
                <div className="space-y-2">
                  {videos.map((video) => (
                    <div key={video.id} className={`flex items-center justify-between p-3 rounded-lg border ${theme === 'dark' ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'}`}>
                      <div className="flex items-center gap-3">
                        <VideoIcon className="w-5 h-5 text-primary" />
                        <div>
                          <p className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{video.name}</p>
                          <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{video.url}</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeVideo(video.id)}
                        className={`p-1 rounded hover:bg-red-500 hover:text-white transition-colors ${theme === 'dark' ? 'text-red-400' : 'text-red-600'}`}
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className={`block text-sm font-medium mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              {t('notes', language)}
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => setFormData({...formData, notes: e.target.value})}
              rows={4}
              className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary outline-none resize-none
                ${theme === 'dark' ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'}`}
              placeholder={language === 'zh-CN' ? '输入宠物的其他信息...' : language === 'zh-HK' ? '輸入寵物的其他資訊...' : 'Enter additional pet information...'}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t">
            <button
              type="button"
              onClick={onClose}
              className={`px-6 py-2 rounded-lg border transition-colors ${
                theme === 'dark' 
                  ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                  : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              {t('cancel', language)}
            </button>
            <button
              type="submit"
              className="gradient-primary text-white px-6 py-2 rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Check className="w-5 h-5" />
              {t('save', language)}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Initial pets data with real video URLs
const initialPets: Pet[] = [
  {
    id: 1,
    name: '小金',
    breed: '金毛寻回犬',
    species: 'dog',
    age: 3,
    weight: 32.5,
    gender: 'male',
    color: '金色',
    imageUrl: 'https://images.unsplash.com/photo-1734966213753-1b361564bab4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxnb2xkZW4lMjByZXRyaWV2ZXIlMjBkb2clMjBwb3J0cmFpdHxlbnwxfHx8fDE3Njk5NDYzNTd8MA&ixlib=rb-4.1.0&q=80&w=1080',
    birthDate: '2023-01-15',
    microchipId: 'MC-2023-001-GR',
    healthStatus: 'healthy',
    lastCheckup: '2026-01-15',
    vaccinated: true,
    photos: [
      'https://images.unsplash.com/photo-1734966213753-1b361564bab4?w=800',
      'https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=800',
      'https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=800',
      'https://images.unsplash.com/photo-1558788353-f76d92427f16?w=800'
    ],
    videos: [
      {
        id: 'golden-1',
        name: '小金在公园玩耍',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
        duration: '0:15'
      },
      {
        id: 'golden-2',
        name: '小金接飞盘',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4',
        duration: '0:15'
      }
    ],
    notes: '性格温顺，喜欢玩飞盘，每天需要至少1小时运动时间。',
    adoptionDate: '2023-03-10'
  },
  {
    id: 2,
    name: 'Luna',
    breed: '英国短毛猫',
    species: 'cat',
    age: 2,
    weight: 4.2,
    gender: 'female',
    color: '蓝灰色',
    imageUrl: 'https://images.unsplash.com/photo-1629624467541-f73ef8f12df2?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxicml0aXNoJTIwc2hvcnRoYWlyJTIwY2F0fGVufDF8fHx8MTc3MDAwNzYxMnww&ixlib=rb-4.1.0&q=80&w=1080',
    birthDate: '2024-03-20',
    microchipId: 'MC-2024-002-BSH',
    healthStatus: 'healthy',
    lastCheckup: '2025-12-20',
    vaccinated: true,
    photos: [
      'https://images.unsplash.com/photo-1629624467541-f73ef8f12df2?w=800',
      'https://images.unsplash.com/photo-1573865526739-10c1dd7adba6?w=800',
      'https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=800'
    ],
    videos: [
      {
        id: 'luna-1',
        name: 'Luna晒太阳',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4',
        duration: '0:15'
      }
    ],
    notes: '安静温柔，喜欢晒太阳，对猫薄荷过敏。',
    adoptionDate: '2024-05-15'
  },
  {
    id: 3,
    name: 'Max',
    breed: '哈士奇',
    species: 'dog',
    age: 4,
    weight: 28.0,
    gender: 'male',
    color: '黑白',
    imageUrl: 'https://images.unsplash.com/photo-1711665722241-a2acd5772864?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzaWJlcmlhbiUyMGh1c2t5JTIwZG9nfGVufDF8fHx8MTc3MDAzODEzM3ww&ixlib=rb-4.1.0&q=80&w=1080',
    birthDate: '2022-06-10',
    microchipId: 'MC-2022-003-HK',
    healthStatus: 'monitoring',
    lastCheckup: '2026-01-28',
    vaccinated: true,
    photos: [
      'https://images.unsplash.com/photo-1711665722241-a2acd5772864?w=800',
      'https://images.unsplash.com/photo-1605568427561-40dd23c2acea?w=800'
    ],
    videos: [
      {
        id: 'max-1',
        name: 'Max在雪地奔跑',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4',
        duration: '0:15'
      },
      {
        id: 'max-2',
        name: 'Max玩雪',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4',
        duration: '0:15'
      },
      {
        id: 'max-3',
        name: 'Max训练课程',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4',
        duration: '0:15'
      }
    ],
    notes: '精力旺盛，需要大量运动。目前在监测关节健康。',
    adoptionDate: '2022-08-01'
  },
  {
    id: 4,
    name: 'Mimi',
    breed: '橘猫',
    species: 'cat',
    age: 1,
    weight: 3.8,
    gender: 'female',
    color: '橘色',
    imageUrl: 'https://images.unsplash.com/photo-1590564472606-4167bce25c20?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0YWJieSUyMGNhdCUyMHBldHxlbnwxfHx8fDE3NzAwMzgxMzN8MA&ixlib=rb-4.1.0&q=80&w=1080',
    birthDate: '2025-02-14',
    microchipId: 'MC-2025-004-TB',
    healthStatus: 'healthy',
    lastCheckup: '2026-01-10',
    vaccinated: true,
    photos: [
      'https://images.unsplash.com/photo-1590564472606-4167bce25c20?w=800',
      'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800'
    ],
    videos: [
      {
        id: 'mimi-1',
        name: 'Mimi追逐玩具',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
        duration: '0:15'
      }
    ],
    notes: '活泼好动，喜欢追逐玩具，食欲旺盛。',
    adoptionDate: '2025-04-20'
  },
  {
    id: 5,
    name: 'Charlie',
    breed: '比格犬',
    species: 'dog',
    age: 5,
    weight: 13.2,
    gender: 'male',
    color: '三色',
    imageUrl: 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxiZWFnbGUlMjBkb2clMjBoYXBweXxlbnwxfHx8fDE3NzAwMzgxMzN8MA&ixlib=rb-4.1.0&q=80&w=1080',
    birthDate: '2021-09-05',
    microchipId: 'MC-2021-005-BG',
    healthStatus: 'healthy',
    lastCheckup: '2025-12-15',
    vaccinated: true,
    photos: [
      'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800',
      'https://images.unsplash.com/photo-1505628346881-b72b27e84530?w=800'
    ],
    videos: [
      {
        id: 'charlie-1',
        name: 'Charlie寻找食物',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
        duration: '0:15'
      }
    ],
    notes: '嗅觉灵敏，爱吃，需要控制体重。',
    adoptionDate: '2021-11-20'
  },
  {
    id: 6,
    name: 'Snow',
    breed: '波斯猫',
    species: 'cat',
    age: 3,
    weight: 5.1,
    gender: 'female',
    color: '纯白色',
    imageUrl: 'https://images.unsplash.com/photo-1735618603118-89e26b0dcf6e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwZXJzaWFuJTIwY2F0JTIwd2hpdGV8ZW58MXx8fHwxNzY5OTk3MzU5fDA&ixlib=rb-4.1.0&q=80&w=1080',
    birthDate: '2023-04-12',
    microchipId: 'MC-2023-006-PS',
    healthStatus: 'healthy',
    lastCheckup: '2026-01-05',
    vaccinated: true,
    photos: [
      'https://images.unsplash.com/photo-1735618603118-89e26b0dcf6e?w=800',
      'https://images.unsplash.com/photo-1536500152107-01ab1422f932?w=800'
    ],
    videos: [
      {
        id: 'snow-1',
        name: 'Snow梳理毛发',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4',
        duration: '0:15'
      },
      {
        id: 'snow-2',
        name: 'Snow优雅散步',
        url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/VolkswagenGTIReview.mp4',
        duration: '0:15'
      }
    ],
    notes: '高贵优雅，需要定期梳理毛发，眼睛需要每天清洁。',
    adoptionDate: '2023-06-01'
  },
];
