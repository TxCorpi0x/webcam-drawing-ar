import ar, hci


def main():
    mp_ar = ar.MasterPiece()

    hand_gs = hci.HandGestureDetector(mp_ar)
    hand_gs.run()


if __name__ == "__main__":
    main()
