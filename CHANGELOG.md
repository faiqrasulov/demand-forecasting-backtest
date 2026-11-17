# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: which results reproduce exactly from the pinned versions (naive, boosting) and which to within 0.15% (daily ETS).

## 2026-11-15

- First version: 12-origin rolling backtest of seasonal-naive, ETS, ridge and gradient boosting with conformal intervals on UCI Bike Sharing.
- README: the hours of the Sandy shutdown are not scored.
