package com.example.wordGram_backend.repository;

import com.example.wordGram_backend.model.Analytics;
import com.example.wordGram_backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface AnalyticsRepository extends JpaRepository<Analytics, Long> {
    Optional<Analytics> findByUserAndDate(User user, LocalDate date);
    List<Analytics> findByUser(User user);
    List<Analytics> findByUserAndDateBetween(User user, LocalDate startDate, LocalDate endDate);
    
    @Query("SELECT a FROM Analytics a WHERE a.user = :user ORDER BY a.date DESC")
    List<Analytics> findByUserOrderByDateDesc(@Param("user") User user);
}

