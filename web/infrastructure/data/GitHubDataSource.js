import { ErrorHandler, fetchWithRetry } from '../error/ErrorHandler.js';

/**
 * GitHub Data Source - Infrastructure Layer
 * Реализация источника данных для GitHub
 */
export class GitHubDataSource {
    constructor(config) {
        this.githubRepo = config.githubRepo;
        this.githubBranch = config.githubBranch;
        this.baseUrl = `https://raw.githubusercontent.com/${this.githubRepo}/${this.githubBranch}`;
    }

    /**
     * Возвращает список кандидатных базовых URL (ветви) для загрузки файлов
     * Приоритет: настроенная ветка → main → master
     */
    _getCandidateBaseUrls() {
        const uniq = new Set();
        const branches = [this.githubBranch, 'main', 'master'];
        const urls = [];
        for (const br of branches) {
            if (!br || uniq.has(br)) continue;
            uniq.add(br);
            urls.push(`https://raw.githubusercontent.com/${this.githubRepo}/${br}`);
        }
        return urls;
    }

    /**
     * Получает данные из index.json
     */
    async fetchData() {
        // Determine language mode based on current page path
        let isEnglish = false;
        try {
            const path = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : '';
            // Treat pages ending with -en.html or _en.html as English mode
            isEnglish = /(?:-|_)en\.html$/i.test(path);
        } catch (e) { /* no-op: default to RU */ }

        // Prefer EN data on EN pages; fallback to alternative EN name, then RU index.json if EN not available
        const fileRU = 'web/infrastructure/data/index.json';
        const fileEN = 'web/infrastructure/data/index-en.json';

        const candidates = this._getCandidateBaseUrls().map(b => `${b}/${isEnglish ? fileEN : fileRU}`);
        // also try alternative EN filename index_en.json if EN
        if (isEnglish) {
            this._getCandidateBaseUrls().forEach(b => candidates.push(`${b}/web/infrastructure/data/index_en.json`));
        }
        let lastErr = null;
        for (const url of candidates) {
            try {
                const resp = await fetchWithRetry(url, {}, 'данные статей');
                if (!resp.ok) continue;
                const data = await resp.json();
                if (!data.years || !Array.isArray(data.years)) continue;
                return data;
            } catch (e) {
                lastErr = e;
                continue;
            }
        }
        throw new Error(`Failed to fetch research data from all branches${lastErr ? `: ${lastErr.message}` : ''}`);
    }

    /**
     * Получает markdown контент обзора с поддержкой новой структуры research/<id>/review/review.md
     * и обратной совместимостью со старым путём /<year>/<weekId>/review.md
     */
    async fetchMarkdown(yearNumber, weekId) {
        let isEnglish = false;
        try {
            const path = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : '';
            isEnglish = /(?:-|_)en\.html$/i.test(path);
        } catch (e) { /* default RU */ }

        const preferred = isEnglish ? 'review-en.md' : 'review.md';
        const altEn = 'review_en.md';

        const baseUrls = this._getCandidateBaseUrls();
        const candidates = [];
        // New structure first
        baseUrls.forEach(b => candidates.push(`${b}/research/${weekId}/review/${preferred}`));
        if (isEnglish) baseUrls.forEach(b => candidates.push(`${b}/research/${weekId}/review/${altEn}`));
        baseUrls.forEach(b => candidates.push(`${b}/research/${weekId}/review/review.md`));
        // Legacy structure
        baseUrls.forEach(b => candidates.push(`${b}/${yearNumber}/${weekId}/${preferred}`));
        if (isEnglish) baseUrls.forEach(b => candidates.push(`${b}/${yearNumber}/${weekId}/${altEn}`));
        baseUrls.forEach(b => candidates.push(`${b}/${yearNumber}/${weekId}/review.md`));

        for (const url of candidates) {
            try {
                const response = await fetchWithRetry(url, {}, `обзор "${weekId}"`);
                if (!response.ok) continue;
                const text = await response.text();
                if (!text || !text.trim()) continue;
                return text;
            } catch (e) {
                continue;
            }
        }
        throw new Error('Не удалось найти обзор ни по одному пути (new/legacy, branches).');
    }

    /**
     * Проверяет доступность GitHub API
     */
    async checkHealth() {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    /**
     * Получает информацию о репозитории
     */
    async getRepositoryInfo() {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}`);
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to fetch repository info: ${error.message}`);
        }
    }

    /**
     * Получает список файлов в директории
     */
    async getDirectoryContents(path) {
        try {
            const response = await fetch(`https://api.github.com/repos/${this.githubRepo}/contents/${path}?ref=${this.githubBranch}`);
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Failed to fetch directory contents: ${error.message}`);
        }
    }
} 
