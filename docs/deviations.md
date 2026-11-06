# Deviations from OpenSees

Every deliberate difference between FemoraX and the OpenSees class it ports, and every
tolerance that is looser than the default. A port is not finished until its deviations are
listed here.

Default tolerances (until a component states otherwise):

- Material strain-path tests: relative 1e-14 on stress and tangent.
- Element tests: relative 1e-12 on force, stiffness and mass.
- System tests: to be set with the first system test, with the reason.

## Entry format

```
### <Component> — <one-line summary>
- OpenSees: <file>, commit <hash>
- Difference: <what FemoraX does differently>
- Why: <reason>
- Effect on results: <none / roundoff / bounded by ...>
- Test: <test that covers it>
```

## Entries

None yet.
