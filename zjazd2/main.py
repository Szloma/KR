import argparse

import tensorflow as tf
import numpy as np


## --points 1,0 0,1 1,1 --angle 90
## --A 2 1 1 3 --b 1 2
## helpers
def convert_points(points):
    return np.array([np.fromstring(p, sep=",") for p in points], dtype=np.float32)


def to_radian(value):
    return value * np.pi / 180


def rotate_around_origin(tensor, angle):
    angle = to_radian(angle)
    rotation_matrix = tf.stack(
        [[tf.cos(angle), -tf.sin(angle)], [tf.sin(angle), tf.cos(angle)]]
    )
    return tf.matmul(tensor, tf.transpose(rotation_matrix))


def solve_linalg(A_vals, b_vals):
    a_size = len(A_vals)
    b_size = len(b_vals)
    n = int(np.sqrt(a_size))
    if n * n != a_size or b_size != n:
        print("--A musi zawierać n^2 elementów, --b musi zawierać n elementów")
        return

    A = tf.constant(np.array(A_vals, dtype=np.float32).reshape(n, n))
    b = tf.constant(np.array(b_vals, dtype=np.float32).reshape((n, 1)))

    if tf.linalg.det(A).numpy() == 0:
        print("Wyznacznik macierzy = 0")
        return

    x = tf.linalg.solve(A, b)
    return x.numpy()#tf.reshape(x, [-1]) ##easier for reading


def main():
    parser = argparse.ArgumentParser(
        description="Obrót macierzy oraz rozwiązywanie równań liniowych."
    )
    parser.add_argument("--angle", type=float, required=False, help="Rotation angle")
    parser.add_argument(
        "--points", nargs="+", required=False, help="ex. --points 1,0 0,1 1,1"
    )

    parser.add_argument(
        "--A",
        nargs="+",
        type=float,
        required=False,
        help="Macierz współczynników",
    )
    parser.add_argument(
        "--b",
        nargs="+",
        type=float,
        required=False,
        help="Wyrazy wolne",
    )

    args = parser.parse_args()

    if args.angle and args.points:
        points = convert_points(args.points)
        rotated_points = rotate_around_origin(tf.constant(points), args.angle)
        print("Obrócona macierz: ")
        print(rotated_points.numpy())

    if args.A and args.b:

        try:
            result = solve_linalg(args.A, args.b)
            print("X= ", result)
        except Exception as e:
            print("Exception when solving linalg: ", e)


if __name__ == "__main__":
    main()
