package com.example.wordGram_backend.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "user_statistics")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserStatistics {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false, unique = true)
    private User user;

    @Column(name = "total_words_checked", columnDefinition = "BIGINT DEFAULT 0")
    private Long totalWordsChecked = 0L;

    @Column(name = "total_spell_checks", columnDefinition = "BIGINT DEFAULT 0")
    private Long totalSpellChecks = 0L;

    @Column(name = "total_grammar_checks", columnDefinition = "BIGINT DEFAULT 0")
    private Long totalGrammarChecks = 0L;

    @Column(name = "total_sessions", columnDefinition = "INT DEFAULT 0")
    private Integer totalSessions = 0;

    @Column(name = "total_time_spent_minutes", columnDefinition = "BIGINT DEFAULT 0")
    private Long totalTimeSpentMinutes = 0L;

    @Column(name = "last_activity_at")
    private LocalDateTime lastActivityAt;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}

