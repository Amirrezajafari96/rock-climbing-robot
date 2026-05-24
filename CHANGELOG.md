# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository scaffold with full modular architecture
- MuJoCo MJCF model: 3-limbed climbing robot with wall and holds
- Forward kinematics via DH-parameter chain
- Inverse kinematics via Jacobian pseudoinverse with null-space projection
- COM stability checker with support polygon and margin computation
- Autonomous climbing planner with A*-guided hold selection
- Simulator abstraction layer (MuJoCo backend + abstract interface)
- Full pytest test suite (unit, integration, simulation)
- GitHub Actions CI workflow
- YAML-based configuration system

## [0.1.0] - 2026-05-23

### Added
- Project created
