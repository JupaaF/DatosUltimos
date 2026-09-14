"""Plantilla mínima para una presentación con Manim Slides.

Renderiza con:
    manim -pqh presentation.py Presentacion

Después presenta el resultado con:
    manim-slides Presentacion
"""

import sys
from pathlib import Path

from manim import *
from manim_slides.slide import Slide, ThreeDSlide

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packing.scene import play_packing

#Color para resaltar: Amarillo apagado
AMARILLO = "#D4AF37"

class Presentacion(ThreeDSlide):
    """Punto de partida para editar tus diapositivas."""

    def construct(self) -> None:

        #Diapositiva 1

        tituloDiapositiva1 = Tex("Redes neuronales ", r"\\para la predicción de la ", r"\\conductividad térmica").scale(2).shift(UP*0.4)
        tituloDiapositiva1[0].set_color(AMARILLO)
        tituloDiapositiva1[2].set_color(AMARILLO)
        nombreDiapositiva1 = Tex("Juan Pablo Fernandez").scale(0.7).to_edge(DOWN+ LEFT)
        self.play(Write(tituloDiapositiva1),Write(nombreDiapositiva1))

        self.wait()
        #Diapositiva 2
        self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        tituloDiapositiva2 = Tex("Outline" ).scale(2).to_edge(UP+LEFT).set_color(AMARILLO)
        self.play(Write(tituloDiapositiva2))

        #TODO: Poner bien los items del outline

        itemsDiapositiva2 = [Tex(item, font_size=60) for item in ["Recapitulación", "Datos", "Metodología", "Resultados", "Conclusiones"]]
        flechasDiapositiva2 = [Tex(r"$\rightarrow$").scale(1).set_color(AMARILLO) for _ in itemsDiapositiva2]
        for i, (item, flecha) in enumerate(zip(itemsDiapositiva2, flechasDiapositiva2)):
            flecha.next_to(tituloDiapositiva2, DOWN, buff=1.1 + i * 1.1)
            flecha.shift(LEFT)
            item.next_to(flecha, RIGHT, buff=0.3)

        self.play(
            *[Write(item) for item in itemsDiapositiva2],
            *[Write(flecha) for flecha in flechasDiapositiva2],
        )

        self.wait()
        #Diapositiva 3
        self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        packingDiapositiva3 = play_packing(self, run_time=10)

        self.wait()
        #Diapositiva 4
        self.next_slide()
        # Izquierda de la pantalla, teniendo en cuenta la orientación 3D.
        izquierda = self.camera.get_rotation_matrix().T @ LEFT
        self.play(packingDiapositiva3.animate.shift(2 * izquierda))




        
