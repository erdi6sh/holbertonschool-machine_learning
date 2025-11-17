# API Integration Projects

This repository contains several Python scripts and modules demonstrating how to interact with different public APIs to retrieve and process data. These projects focus on practical usage of RESTful APIs including pagination handling, parsing JSON responses, and dealing with API-specific constraints such as rate limits.

## Projects Overview

### 0. Available Ships (Swapi API)
- **Description:** A Python method that queries the Star Wars API (Swapi) to return a list of starships capable of holding at least a specified number of passengers.
- **Function:** `availableShips(passengerCount)`
- **Features:** 
  - Supports pagination to ensure retrieval of all starship data.
  - Filters starships by their passenger capacity.
  - Returns an empty list if no suitable ship is found.
- **Sample Output:** 
