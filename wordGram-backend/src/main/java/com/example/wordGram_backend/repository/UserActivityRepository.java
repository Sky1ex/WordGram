package com.example.wordGram_backend.repository;

import com.example.wordGram_backend.model.User;
import com.example.wordGram_backend.model.UserActivity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface UserActivityRepository extends JpaRepository<UserActivity, Long> {
    List<UserActivity> findByUser(User user);
    List<UserActivity> findByUserAndCreatedAtBetween(User user, LocalDateTime start, LocalDateTime end);
    List<UserActivity> findByUserOrderByCreatedAtDesc(User user);
}

