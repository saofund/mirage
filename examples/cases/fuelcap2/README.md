# Fuelcap 2: 2007 Polo

An independent photograph match for a circular, metallic-blue filler region. It does not
reuse case 27's white BoYue hero parameters; only the tested low-level part constructors
are shared.

```powershell
python -m fuelcap2.sheet render --size 1000 --spp 160
python -m fuelcap2.sheet compose --size 1000
```

`render` is intended for the remote build box. `compose` runs locally because the ignored
customer reference set is intentionally absent from the box.
