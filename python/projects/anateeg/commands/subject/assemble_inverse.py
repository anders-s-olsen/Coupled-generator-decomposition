import sys

from projects.anateeg import inverse
from projects.anateeg.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    inverse.assemble_inverse(subject_id)