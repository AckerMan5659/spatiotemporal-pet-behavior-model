export const API_BASE_URL = 'http://127.0.0.1:9527';

export interface BackendStats {
    [camId: string]: {
        stats: {
            fps: number;
            yoloMs: number;
            ruleMs: number;
            actionMs: number;
            decodeMs: number;
            status: string;
            mode: string;
            imgsz?: number;
            isRecording?: boolean;
            camId: number;
            name?: string; // 🔥 新增：接收後端動態提取的名字
        };
        logs: any[];
        active_states: {
            [trackId: string]: {
                id: number;
                time: string;
                bowl: string;
                probs: Record<string, number>;
            };
        };
    };
}

export interface VideoRecord {
    id: number;
    cam_id: number;
    filename: string;
    start_time: string;
    end_time: string;
    trigger_action: string;
    max_confidence: number;
}

export const fetchCameraStats = async (): Promise<BackendStats> => {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch camera stats:', error);
        return {};
    }
};

export const getActiveCameras = async (): Promise<number[]> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/active_cams`);
        if (!response.ok) throw new Error('Network response not ok');
        const data = await response.json();
        return data.active_cams || [];
    } catch (error) {
        console.error('Failed to get active cameras:', error);
        return [];
    }
};

export const setActiveCameras = async (camIds: number[]): Promise<boolean> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/active_cams`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ active_cams: camIds }),
        });
        return response.ok;
    } catch (error) {
        console.error('Failed to set active cameras:', error);
        return false;
    }
};

export const getCameraConfig = async (camId: number) => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/config/${camId}`);
        if (!response.ok) throw new Error('Failed to fetch config');
        return await response.json();
    } catch (error) {
        console.error(`Failed to get config for CAM-${camId}:`, error);
        return null;
    }
};

export const updateCameraConfig = async (camId: number, config: any): Promise<boolean> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/config/${camId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        return response.ok;
    } catch (error) {
        console.error(`Failed to update config for CAM-${camId}:`, error);
        return false;
    }
};

export const getVideoRecords = async (filters: { cam_id?: string; action?: string; date?: string }): Promise<VideoRecord[]> => {
    try {
        const queryParams = new URLSearchParams();
        if (filters.cam_id && filters.cam_id !== 'all') queryParams.append('cam_id', filters.cam_id);
        if (filters.action && filters.action !== 'all') queryParams.append('action', filters.action);
        if (filters.date) queryParams.append('date', filters.date);

        const response = await fetch(`${API_BASE_URL}/api/records?${queryParams.toString()}`);
        if (!response.ok) throw new Error('Failed to fetch records');
        const data = await response.json();
        return data.success ? data.records : [];
    } catch (error) {
        console.error('Failed to get video records:', error);
        return [];
    }
};

export const deleteVideoRecord = async (id: number): Promise<boolean> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/records/${id}`, {
            method: 'DELETE',
        });
        return response.ok;
    } catch (error) {
        console.error('Failed to delete video record:', error);
        return false;
    }
};

export const getRecordVideoUrl = (filename: string) => {
    const cleanName = filename.replace('records/', '').replace('records\\', '');
    return `${API_BASE_URL}/records/${cleanName}`;
};

export const getRecordThumbnailUrl = (filename: string) => {
    const cleanName = filename
        .replace('records/', '')
        .replace('records\\', '')
        .replace('/videos/', '/thumbnails/')
        .replace('\\videos\\', '\\thumbnails\\')
        .replace('.webm', '.jpg')
        .replace('.mp4', '.jpg');
    return `${API_BASE_URL}/records/${cleanName}`;
};

export const getVideoFeedUrl = (camId: number | string) => {
    let index = typeof camId === 'string' && camId.startsWith('CAM-')
        ? parseInt(camId.split('-')[1]) - 1
        : Number(camId);

    if (isNaN(index)) index = 0;

    try {
        const url = new URL(API_BASE_URL);
        if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') {
            const shardedHost = `127.0.0.${(index % 8) + 1}`;
            return `http://${shardedHost}:${url.port}/video_feed/${index}`;
        }
    } catch (e) {
        console.error(e);
    }
    return `${API_BASE_URL}/video_feed/${index}`;
};