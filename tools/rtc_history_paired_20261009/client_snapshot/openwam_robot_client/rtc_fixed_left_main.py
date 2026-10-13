"""Independent TrainingRTC entry: fixed left arm/hand, control right side only."""
from __future__ import annotations

from functools import partial
import logging
import signal

from .emergency_left_thumb import RIGHT_USB_SERIAL
from .rtc_fixed_left import DEFAULT_INIT_FILE, FixedLeftController, FixedLeftHands, read_fixed_left_init
from .rtc_main import parse_args as rtc_parse_args
from .rtc_runtime import run


def parse_args(argv=None):
    def configure(parser):
        parser.description = 'OpenWAM TrainingRTC: fixed left arm/hand, right-side control only'
        parser.add_argument('--fixed-left-init-file', default=str(DEFAULT_INIT_FILE),
                            help='JSON with torso(4), left arm(7), and left hand(20) initial joints')
        parser.add_argument('--rtc-shadow-align', action='store_true',
                            help='Record metadata for passive alignment analysis; never changes inference inputs')
        parser.add_argument('--rtc-eef-filter-alpha', type=float, default=1.0,
                            help='Optional cross-chunk EEF target filter: 1 disables (original behavior); '
                                 '0.3 filters XYZ and quaternion rotation before constraints and prefix commitment')
        parser.set_defaults(hold_left_init=True, right_hand_serial=RIGHT_USB_SERIAL)
    args = rtc_parse_args(argv, configure_parser=configure)
    if args.ignore_left_thumb or args.use_wbc or args.move_home or args.left_hand_serial:
        raise ValueError('Fixed-left client forbids --ignore-left-thumb, --use-wbc, --move-home and --left-hand-serial')
    if args.right_hand_serial != RIGHT_USB_SERIAL:
        raise ValueError(f'Expected verified right USB serial {RIGHT_USB_SERIAL}')
    args.fixed_left_mode = True
    if not 0 < args.rtc_eef_filter_alpha <= 1:
        raise ValueError('--rtc-eef-filter-alpha must be finite and in (0, 1]')
    if args.rtc_shadow_align and not (args.rtc_state_trace and (args.rtc_diagnostics_dir or args.rtc_diagnostics_file)):
        raise ValueError('--rtc-shadow-align requires diagnostics and --rtc-state-trace')
    return args


def main(argv=None):
    args = parse_args(argv)
    initial = read_fixed_left_init(args.fixed_left_init_file)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger(__name__).warning(
        'Fixed-left mode: left hand disconnected; left EEF/hand state and prefixes fixed; only right-side targets are sent')
    controller_factory = FixedLeftController
    if args.rtc_eef_filter_alpha < 1.0:
        from .rtc_eef_filter import EEFFilteredFixedLeftController
        controller_factory = partial(EEFFilteredFixedLeftController,
                                     eef_filter_alpha=args.rtc_eef_filter_alpha)
        logging.getLogger(__name__).warning(
            'Client EEF filter enabled: alpha=%s, XYZ + quaternion SLERP, continuous across chunks; '
            'filtered committed targets become RTC prefixes. SDK control_way=%s filter_scale=%s remains unchanged.',
            args.rtc_eef_filter_alpha, args.control_way, args.filter_scale)
    def stop(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        run(args, hands_factory=partial(FixedLeftHands, fixed_left_hand=initial[2]),
            controller_factory=controller_factory, left_init_loader=lambda: initial)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info('RTC fixed-left client stopped')


if __name__ == '__main__':
    main()
