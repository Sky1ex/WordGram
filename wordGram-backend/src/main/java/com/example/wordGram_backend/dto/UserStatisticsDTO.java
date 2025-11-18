package com.example.wordGram_backend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserStatisticsDTO {
    private Long id;
    private Long userId;
    private Long totalWordsChecked;
    private Long totalSpellChecks;
    private Long totalGrammarChecks;
    private Integer totalSessions;
    private Long totalTimeSpentMinutes;
    private LocalDateTime lastActivityAt;
}

