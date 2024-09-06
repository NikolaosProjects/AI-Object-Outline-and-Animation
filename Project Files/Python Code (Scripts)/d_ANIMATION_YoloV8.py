from c_FOURIER_YoloV8 import *

angles = []

config.disable_caching = True

class AIOutline(Scene):
    def construct(self):
        # Clear previous traces if any
        self.clear()

        # Define and create all vectors
        vectors = []
        for i in range(len(data)):
            vector = Vector(data[i]/zoomval, color=WHITE)
            vector.set_stroke(width=2.5)
            vector.rotate(-PI/2)
            if i > 0:
                vector.shift(vectors[i-1].get_end())
            vectors.append(vector)

        # Add all vectors to the scene at once
        self.play(*[Create(vec) for vec in vectors], run_time=1.e-10)

        # Define the rotation rate for the vectors (ω = 2πf)
        for i in range(len(f)):
            angles.append(500 * 2 * np.pi * f[i])  # Different rates of rotation

        # Create traced path for the last vector's tip
        traced_path = VMobject(stroke_color=YELLOW, stroke_width=2.5)
        self.add(traced_path)

        # Create updater functions for each vector
        def create_vector_updater(vector, previous_vector, angle):
            def update_vector(vec, dt):
                vec.shift(previous_vector.get_end() - vec.get_start())
                vec.rotate(angle * dt, about_point=previous_vector.get_end())
            return update_vector

        # Apply updaters to all vectors except the last one
        for i in range(len(data)):
            if i == 0:
                # The first vector rotates around the origin
                vectors[i].add_updater(lambda vec, dt, angle=angles[i]: vec.rotate(angle * dt, about_point=ORIGIN))
            else:
                # Subsequent vectors rotate around the tip of the previous vector
                updater = create_vector_updater(vectors[i], vectors[i-1], angles[i])
                vectors[i].add_updater(updater)

        # Create updater function for the last vector to trace its path
        def update_last_vector(vec, dt):
            vec.shift(vectors[-2].get_end() - vec.get_start())
            vec.rotate(angles[-1] * dt, about_point=vectors[-2].get_end())
            if len(traced_path.points) > 0:
                traced_path.add_line_to(vec.get_end())
            else:
                traced_path.start_new_path(vec.get_end())

        # Apply updater to the last vector
        vectors[-1].add_updater(update_last_vector)

        # Run the animation
        self.wait(anim_dur)  # Wait for the animation to complete

        # Stop all updaters to freeze the final frame
        for vec in vectors:
            vec.clear_updaters()

        # Fade out vectors while keeping the trace visible
        self.play(*[FadeOut(vec) for vec in vectors], run_time=0.5)

        # Keep the traced path on the screen and pause
        self.wait(1.5)

        # Clear everything from the scene after the pause
        self.clear()
