package com.example.wordGram_backend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AnalyticsDTO {
    private Long id;
    private Long userId;
    private LocalDate date;
    private Integer wordsChecked;
    private Integer spellChecks;
    private Integer grammarChecks;
    private Integer sessionsCount;
    private Integer timeSpentMinutes;
}

