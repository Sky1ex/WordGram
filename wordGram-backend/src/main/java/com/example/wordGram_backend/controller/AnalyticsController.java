package com.example.wordGram_backend.controller;

import com.example.wordGram_backend.dto.AnalyticsDTO;
import com.example.wordGram_backend.dto.UserStatisticsDTO;
import com.example.wordGram_backend.model.UserActivity;
import com.example.wordGram_backend.service.AnalyticsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@RestController
@RequestMapping("/api/analytics")
@CrossOrigin(origins = "*")
public class AnalyticsController {

    @Autowired
    private AnalyticsService analyticsService;

    @PostMapping("/activity")
    public ResponseEntity<String> recordActivity(
            @RequestParam Long userId,
            @RequestParam UserActivity.ActivityType activityType,
            @RequestParam(required = false) String description,
            @RequestParam(required = false) String metadata) {
        try {
            analyticsService.recordActivity(userId, activityType, description, metadata);
            return ResponseEntity.ok("Activity recorded successfully");
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    @GetMapping("/users/{userId}/statistics")
    public ResponseEntity<UserStatisticsDTO> getUserStatistics(@PathVariable Long userId) {
        try {
            UserStatisticsDTO statistics = analyticsService.getUserStatistics(userId);
            return ResponseEntity.ok(statistics);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @GetMapping("/users/{userId}/daily")
    public ResponseEntity<AnalyticsDTO> getDailyAnalytics(
            @PathVariable Long userId,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date) {
        try {
            AnalyticsDTO analytics = analyticsService.getDailyAnalytics(userId, date);
            return ResponseEntity.ok(analytics);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @GetMapping("/users/{userId}")
    public ResponseEntity<List<AnalyticsDTO>> getUserAnalytics(
            @PathVariable Long userId,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
        try {
            List<AnalyticsDTO> analytics = analyticsService.getUserAnalytics(userId, startDate, endDate);
            return ResponseEntity.ok(analytics);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @PostMapping("/users/{userId}/time")
    public ResponseEntity<String> updateTimeSpent(
            @PathVariable Long userId,
            @RequestParam Integer minutes) {
        try {
            analyticsService.updateTimeSpent(userId, minutes);
            return ResponseEntity.ok("Time spent updated successfully");
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }
}

