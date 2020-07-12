import sys
import time

class ProgressBar(object):

    def __init__(self, max, print_interval=5):
        self.max = max
        self.print_interval = print_interval
        self.last_print = time.time() - print_interval
        self.start_time = time.time()

        self.update(0)


    def update(self, progress, message=""):
        if progress < self.max - 1 and time.time() - self.last_print < self.print_interval:
            return

        self.last_print = time.time()

        BAR_LENGTH = 20
        SYMBOL = '-'

        ratio = float(progress + 1) / float(self.max)

        length = int(ratio * BAR_LENGTH)

        bar = (SYMBOL * length) + (' ' * (BAR_LENGTH - length))

        elapsed = (time.time() - self.start_time)

        eta = (elapsed / (progress + 1)) * (self.max - progress - 1)

        elapsed_str = time2str(elapsed)
        eta_str = time2str(eta)

        sys.stdout.write("\rSTART|%s|END. %d of %d. Elapsed: %s, ETA: %s %s\n" %
                         (bar, progress+1, self.max, elapsed_str, eta_str, message))
        sys.stdout.flush()

        if progress == self.max - 1:
            print("\nDone")



def time2str(t):
    if t > 3600:
        return  "%1.2f hr." % (t / 3600.0)
    elif t > 60:
        return "%1.2f min." % (t /60.0)
    else:
        return "%1.2f sec." % t


if __name__ == "__main__":

    n = 5

    bar = ProgressBar(n)

    for i in range(n):
        time.sleep(1)
        bar.update(i)

