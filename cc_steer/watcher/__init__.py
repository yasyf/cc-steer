"""The live watcher runtime: tail transcripts, run the steering cascade in shadow mode.

The offline pipeline mines how this user steers; this package acts on it live.
A daemon tails every open Claude Code session, and each time a session goes
quiet after completing a turn, a three-stage cascade decides whether the user
would have stepped in right there: a cheap gate scores the moment, a drafting
model writes the steer the user would send, and a frontier refiner — shown how
the user steered in similar past moments — issues the final message or
abstains. Every verdict lands in the shadow ledger; nothing reaches a live
session until shadow metrics prove the cascade sendable.
"""
