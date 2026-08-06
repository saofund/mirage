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

`photo_maps.py` bakes four ignored projective albedo maps from that one reference. The
cap, liner and door maps preserve local wear; the calibrated panel projection preserves
the photographed clear-coat reflection and body-colour variation while the actual 3D
opening and parts supply silhouette, depth and occlusion. Copy
`assets/decals/fuelcap_polo_*_photo.ppm` to a render host together with the case source.
