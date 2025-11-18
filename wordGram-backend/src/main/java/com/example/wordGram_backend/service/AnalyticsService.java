package com.example.wordGram_backend.service;

import com.example.wordGram_backend.dto.AnalyticsDTO;
import com.example.wordGram_backend.dto.UserStatisticsDTO;
import com.example.wordGram_backend.model.Analytics;
import com.example.wordGram_backend.model.User;
import com.example.wordGram_backend.model.UserActivity;
import com.example.wordGram_backend.model.UserStatistics;
import com.example.wordGram_backend.repository.AnalyticsRepository;
import com.example.wordGram_backend.repository.UserActivityRepository;
import com.example.wordGram_backend.repository.UserRepository;
import com.example.wordGram_backend.repository.UserStatisticsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AnalyticsService {

    @Autowired
    private AnalyticsRepository analyticsRepository;

    @Autowired
    private UserStatisticsRepository userStatisticsRepository;

    @Autowired
    private UserActivityRepository userActivityRepository;

    @Autowired
    private UserRepository userRepository;

    @Transactional
    public void recordActivity(Long userId, UserActivity.ActivityType activityType, String description, String metadata) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        UserActivity activity = new UserActivity();
        activity.setUser(user);
        activity.setActivityType(activityType);
        activity.setDescription(description);
        activity.setMetadata(metadata);
        userActivityRepository.save(activity);

        // Update user statistics
        updateUserStatistics(user, activityType);

        // Update daily analytics
        updateDailyAnalytics(user, activityType);
    }

    @Transactional
    private void updateUserStatistics(User user, UserActivity.ActivityType activityType) {
        UserStatistics statistics = userStatisticsRepository.findByUser(user)
                .orElseGet(() -> {
                    UserStatistics newStats = new UserStatistics();
                    newStats.setUser(user);
                    return newStats;
                });

        switch (activityType) {
            case WORD_CHECK -> statistics.setTotalWordsChecked(statistics.getTotalWordsChecked() + 1);
            case SPELL_CHECK -> statistics.setTotalSpellChecks(statistics.getTotalSpellChecks() + 1);
            case GRAMMAR_CHECK -> statistics.setTotalGrammarChecks(statistics.getTotalGrammarChecks() + 1);
            case SESSION_START -> statistics.setTotalSessions(statistics.getTotalSessions() + 1);
            case LOGIN, LOGOUT, SESSION_END, PROFILE_UPDATE, OTHER -> {
                // These activities don't update statistics counters
            }
        }

        statistics.setLastActivityAt(LocalDateTime.now());
        userStatisticsRepository.save(statistics);
    }

    @Transactional
    private void updateDailyAnalytics(User user, UserActivity.ActivityType activityType) {
        LocalDate today = LocalDate.now();
        Analytics analytics = analyticsRepository.findByUserAndDate(user, today)
                .orElseGet(() -> {
                    Analytics newAnalytics = new Analytics();
                    newAnalytics.setUser(user);
                    newAnalytics.setDate(today);
                    return newAnalytics;
                });

        switch (activityType) {
            case WORD_CHECK -> analytics.setWordsChecked(analytics.getWordsChecked() + 1);
            case SPELL_CHECK -> analytics.setSpellChecks(analytics.getSpellChecks() + 1);
            case GRAMMAR_CHECK -> analytics.setGrammarChecks(analytics.getGrammarChecks() + 1);
            case SESSION_START -> analytics.setSessionsCount(analytics.getSessionsCount() + 1);
            case LOGIN, LOGOUT, SESSION_END, PROFILE_UPDATE, OTHER -> {
                // These activities don't update daily analytics counters
            }
        }

        analyticsRepository.save(analytics);
    }

    public UserStatisticsDTO getUserStatistics(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        UserStatistics statistics = userStatisticsRepository.findByUser(user)
                .orElseGet(() -> {
                    UserStatistics newStats = new UserStatistics();
                    newStats.setUser(user);
                    return userStatisticsRepository.save(newStats);
                });

        return convertToStatisticsDTO(statistics);
    }

    public List<AnalyticsDTO> getUserAnalytics(Long userId, LocalDate startDate, LocalDate endDate) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        List<Analytics> analytics;
        if (startDate != null && endDate != null) {
            analytics = analyticsRepository.findByUserAndDateBetween(user, startDate, endDate);
        } else {
            analytics = analyticsRepository.findByUser(user);
        }

        return analytics.stream()
                .map(this::convertToAnalyticsDTO)
                .collect(Collectors.toList());
    }

    public AnalyticsDTO getDailyAnalytics(Long userId, LocalDate date) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        LocalDate targetDate = date != null ? date : LocalDate.now();
        Analytics analytics = analyticsRepository.findByUserAndDate(user, targetDate)
                .orElseGet(() -> {
                    Analytics newAnalytics = new Analytics();
                    newAnalytics.setUser(user);
                    newAnalytics.setDate(targetDate);
                    return analyticsRepository.save(newAnalytics);
                });

        return convertToAnalyticsDTO(analytics);
    }

    @Transactional
    public void updateTimeSpent(Long userId, Integer minutes) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Update statistics
        UserStatistics statistics = userStatisticsRepository.findByUser(user)
                .orElseGet(() -> {
                    UserStatistics newStats = new UserStatistics();
                    newStats.setUser(user);
                    return newStats;
                });
        statistics.setTotalTimeSpentMinutes(statistics.getTotalTimeSpentMinutes() + minutes);
        userStatisticsRepository.save(statistics);

        // Update daily analytics
        LocalDate today = LocalDate.now();
        Analytics analytics = analyticsRepository.findByUserAndDate(user, today)
                .orElseGet(() -> {
                    Analytics newAnalytics = new Analytics();
                    newAnalytics.setUser(user);
                    newAnalytics.setDate(today);
                    return newAnalytics;
                });
        analytics.setTimeSpentMinutes(analytics.getTimeSpentMinutes() + minutes);
        analyticsRepository.save(analytics);
    }

    private AnalyticsDTO convertToAnalyticsDTO(Analytics analytics) {
        AnalyticsDTO dto = new AnalyticsDTO();
        dto.setId(analytics.getId());
        dto.setUserId(analytics.getUser().getId());
        dto.setDate(analytics.getDate());
        dto.setWordsChecked(analytics.getWordsChecked());
        dto.setSpellChecks(analytics.getSpellChecks());
        dto.setGrammarChecks(analytics.getGrammarChecks());
        dto.setSessionsCount(analytics.getSessionsCount());
        dto.setTimeSpentMinutes(analytics.getTimeSpentMinutes());
        return dto;
    }

    private UserStatisticsDTO convertToStatisticsDTO(UserStatistics statistics) {
        UserStatisticsDTO dto = new UserStatisticsDTO();
        dto.setId(statistics.getId());
        dto.setUserId(statistics.getUser().getId());
        dto.setTotalWordsChecked(statistics.getTotalWordsChecked());
        dto.setTotalSpellChecks(statistics.getTotalSpellChecks());
        dto.setTotalGrammarChecks(statistics.getTotalGrammarChecks());
        dto.setTotalSessions(statistics.getTotalSessions());
        dto.setTotalTimeSpentMinutes(statistics.getTotalTimeSpentMinutes());
        dto.setLastActivityAt(statistics.getLastActivityAt());
        return dto;
    }
}

