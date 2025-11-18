package com.example.wordGram_backend.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "analytics", indexes = {
    @Index(name = "idx_user_date", columnList = "user_id,date"),
    @Index(name = "idx_date", columnList = "date")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Analytics {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "date", nullable = false)
    private LocalDate date;

    @Column(name = "words_checked", columnDefinition = "INT DEFAULT 0")
    private Integer wordsChecked = 0;

    @Column(name = "spell_checks", columnDefinition = "INT DEFAULT 0")
    private Integer spellChecks = 0;

    @Column(name = "grammar_checks", columnDefinition = "INT DEFAULT 0")
    private Integer grammarChecks = 0;

    @Column(name = "sessions_count", columnDefinition = "INT DEFAULT 0")
    private Integer sessionsCount = 0;

    @Column(name = "time_spent_minutes", columnDefinition = "INT DEFAULT 0")
    private Integer timeSpentMinutes = 0;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (date == null) {
            date = LocalDate.now();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}

