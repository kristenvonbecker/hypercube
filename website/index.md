---
layout: default
title: Home
---

<figure class="hero-image-container">
  <img 
    src="{{ '/assets/images/zoetrope.png' | relative_url }}" 
    alt="A 3D render of the Hypercube zoetrope design.">
  <figcaption>A rendering of the current zoetrope design, created in Fusion.</figcaption>
</figure>

## The Concept 

How can we make sense of mathematically-defined objects which live beyond the limits of our three-dimensional universe?
*Hypercube* explores this question through the precision-engineered stroboscopic animation of various lower-dimensional
representations of higher-dimensional "cubes". This one-of-a-kind kinematic art installation challenges rigid spatial 
intuition and illustrates the beauty inherent in abstract mathematics. 

*Hypercube* is being developed for the Mathematical Art Exhibit at the 2027 
<a href="https://jointmathematicsmeetings.org/jmm" target="_blank" rel="noopener">Joint Mathematics Meetings</a>, 
and is partially funded by a grant from the National Science Foundation.

---

## The Exhibit

*Hypercube* is a stroboscopic 3D zoetrope --- a device that creates an optical illusion of motion by synchronizing the 
rotation of a sequence of static physical models with the frequency of a flashing strobe light. 
Its subject is the four-dimensional hypercube, also known as the tesseract or 4-cube.

A tesseract is the 4D version of a 3D cube, related to the cube in the same way that a cube is to a square. 
The key to understanding its geometry lies in observing how its 3D representations --- such as projections, 
cross-sections, or nets --- transform as the hypercube is rotated or moved through a higher dimension.

This dynamic transformation is precisely what the zoetrope is designed to illustrate. By sequencing a range of  
static 3D representations and animating them stroboscopically, *Hypercube* builds intuition around higher-dimensional 
geometry and makes an otherwise abstract mathematical construction a tangible reality.

## Lower-Dimensional Representations

To study a higher-dimensional object, we can consider its representations in our own more familiar 3-dimensional space. *Hypercube* animates 
three such lower-dimensional views:

- **Projection** is the mathematical process that describes a shadow. In the same way that a 3-cube casts a
2-dimensional shadow, the projection of a 4-cube (hypercube) is a 3-dimensional object which can be 3D modeled. 
The zoetrope rendering above shows a sequence of the tesseract's "shadows" as it performs a so-called double 
rotation in 4-dimensional space.

- **Cross-sections** are lower-dimensional representations of an object formed by "slicing" the object with a *hyperplane* 
(the higher-dimensional analog of a plane). This sort of investigaton is similar to how an MRI procedure captures a 
sequence of 2-dimensional images which are stacked to build up a 3-dimensionalal understanding of the anatomy.

- **Nets** are obtained by "opening" the boundary of an object. Just like the net of a 3-cube is a 
2-dimensional arrangement of squares, the net of a 4-cube is 3-dimensional arangement of cells.

Through its animation of these three distinct views, *Hypercube* offers a dynamic and multi-faceted --- no pun intended --- 
exploration of a single unseeable object.

## Project Status

The prototype design phase for *Hypercube* is nearing completion. While the primary zoetrope structure is being modeled with CAD 
using Fusion 360 (and is still a work in progress), the SCAD or STL models for the zoetrope's 3D printed figures can now be 
generated through a custom-developed Python + OpenSCAD scripted workflow, which is available and documented in the project's 
<a href="https://github.com/kristenvonbecker/hypercube" target="_blank" rel="noopener">GitHub repository</a>.

*Hypercube*'s fabrication phase is scheduled to begin soon. All (manual and CNC) woodworking steps will be completed at the 
<a href="https://sdfwa.org" target="_blank" rel="noopener">San Diego Fine Woodworkers 
Association</a> member shop. After the physical construction of the zoetrope's main rotational assembly prototype is complete, 
the Arduino-based control system and necessary electronics will be integrated before the prototype can be tested and 
improved upon for the final version.

Please check back later (or subscribe to the [Hyperblog](/blog.md)!) for information about project updates.

---

## Latest Updates

<ul>
  {% for post in site.posts limit:3 %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a> - <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time>
    </li>
  {% else %}
    <li>No posts found yet.</li>
  {% endfor %}
</ul>
