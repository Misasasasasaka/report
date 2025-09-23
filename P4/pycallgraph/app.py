"""A tiny demo program to generate a call graph with pycallgraph2.

Run directly:
    python app.py

Or via trace.py to generate callgraph.png:
    python trace.py
"""

import time

class Greeter:
    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"

def slow_double(x: int) -> int:
    time.sleep(0.1)  # simulate work
    return x * 2

def compute_score(nums):
    total = 0
    for n in nums:
        total += slow_double(n)
    return total

def main():
    user = Greeter("Misaka")
    print(user.greet())
    nums = [1, 2, 3, 4]
    score = compute_score(nums)
    print("Score:", score)

if __name__ == "__main__":
    main()
