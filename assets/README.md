# Exact-Front Textured 3D Relief Models

Each model needs three files kept in the same folder:

- `dragon.obj`, `dragon.mtl`, `dragon_texture.png`
- `tree.obj`, `tree.mtl`, `tree_texture.png`
- `flowers.obj`, `flowers.mtl`, `flowers_texture.png`
- `butterfly.obj`, `butterfly.mtl`, `butterfly_texture.png`

These assets preserve the supplied PNG artwork as the exact front/back texture
and add a shaped relief surface plus solid side thickness. They are 2.5D relief
models, not complete unseen-side anatomical reconstructions. An OBJ loader must
support UV coordinates, MTL `map_Kd`, PNG alpha, and alpha blending for the
transparent silhouette to render correctly.
