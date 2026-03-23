class PadelScoringEngine:
    def __init__(self, golden_point=True):
        self.scores = ["0", "15", "30", "40", "AD"]
        self.team1_points = 0
        self.team2_points = 0
        self.team1_games = 0
        self.team2_games = 0
        self.team1_sets = 0
        self.team2_sets = 0
        self.golden_point = golden_point
        self.game_over = False

    def get_score_display(self):
        t1_pts = self.scores[self.team1_points]
        t2_pts = self.scores[self.team2_points]
        return {
            "score": f"{t1_pts} - {t2_pts}",
            "games": f"{self.team1_games} - {self.team2_games}",
            "sets": f"{self.team1_sets} - {self.team2_sets}"
        }

    def add_point(self, team_id):
        if self.game_over:
            return

        if team_id == 1:
            self._increment_point(1)
        elif team_id == 2:
            self._increment_point(2)

    def _increment_point(self, team_id):
        other_team = 2 if team_id == 1 else 1
        
        # Current points of teams
        p1 = self.team1_points if team_id == 1 else self.team2_points
        p2 = self.team2_points if team_id == 1 else self.team1_points

        # Logic for Golden Point (Punto de Oro)
        if self.golden_point and p1 == 3 and p2 == 3:
            self._win_game(team_id)
            return

        # Regular Scoring Logic
        if p1 < 3:
            if team_id == 1: self.team1_points += 1
            else: self.team2_points += 1
        elif p1 == 3: # At 40
            if p2 < 3: # Other team has < 40
                self._win_game(team_id)
            elif p2 == 3: # Deuce
                if team_id == 1: self.team1_points = 4 # AD
                else: self.team2_points = 4 # AD
            elif p2 == 4: # Other team had AD, now back to Deuce
                if team_id == 1: self.team2_points = 3
                else: self.team1_points = 3
        elif p1 == 4: # At AD
            self._win_game(team_id)

    def _win_game(self, team_id):
        if team_id == 1:
            self.team1_games += 1
        else:
            self.team2_games += 1
        
        self.team1_points = 0
        self.team2_points = 0
        
        self._check_set(team_id)

    def _check_set(self, team_id):
        g1, g2 = self.team1_games, self.team2_games
        
        if (g1 >= 6 and g1 - g2 >= 2) or g1 == 7:
            self.team1_sets += 1
            self.team1_games = 0
            self.team2_games = 0
        elif (g2 >= 6 and g2 - g1 >= 2) or g2 == 7:
            self.team2_sets += 1
            self.team1_games = 0
            self.team2_games = 0

        # Check Match Win
        if self.team1_sets == 2 or self.team2_sets == 2:
            self.game_over = True

# Example Usage:
# engine = PadelScoringEngine()
# engine.add_point(1) # Team 1 wins a point
# print(engine.get_score_display())
