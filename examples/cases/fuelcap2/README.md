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

The materials are **procedural**. An earlier version of this case baked four albedo maps
straight out of the reference photograph — the panel, the cap, the liner and the door
inner — and projected them back onto the geometry. It looked convincing because it largely
*was* the photograph, and that is the objection: it cannot show that the model is right,
it smears wherever the geometry disagrees with the picture, and it must never reach a
training set, because a network would learn the reference frames themselves. Removed.

What replaces it is `textures._beaded_water`. The car was photographed in the rain and the
beading is most of the surface character: a few hundred tiny near-mirrors on a matt black
shell, each returning a hard white point where the substrate around it returns almost
nothing. That is a roughness-and-normal effect, not an albedo one — painting light dots
into a colour map reads as dirt. The larger beads on the cap and the door are real
geometry; the fine ones are the map.
