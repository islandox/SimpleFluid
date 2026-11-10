"""Compare replay summaries and attach diagnostic-only owned-cell evidence."""

import csv
import json
import math
from fractions import Fraction
from pathlib import Path


RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-10
NORMS = ('volume_rms', 'max_density', 'signed_sum')
COMPONENTS = ('k', 'secondary', 'reconstructed_continuity',
              'momentum_x', 'momentum_y', 'momentum_z')


def _coordinate_key(coordinates):
    # Match PitzDailyReplay.hh's std::llround, including halfway-away-from-zero.
    key = []
    for coordinate in coordinates:
        scaled = coordinate * 1e12
        if not math.isfinite(scaled) or abs(scaled) >= 9e18:
            raise ValueError('Replay coordinate cannot be indexed')
        magnitude = abs(scaled)
        integer = math.floor(magnitude)
        rounded = integer + (magnitude - integer >= 0.5)
        key.append(rounded if scaled >= 0 else -rounded)
    return tuple(key)


def _finite_number(value, description):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{description} must be finite')
    return number


def _read_cells(directory):
    paths = sorted(Path(directory).glob('diagnostics_step0_cells_rank*.csv'))
    if not paths:
        raise FileNotFoundError(f'No diagnostics_step0_cells_rank*.csv in {directory}')
    cells = []
    geometry_columns = set()
    seen_ids = set()
    seen_coordinates = set()
    seen_geometry_ids = set()
    density_columns = ('momentum_x', 'momentum_y', 'momentum_z', 'continuity', 'k', 'secondary')
    required = {'cell_id', 'x', 'y', 'z', 'volume', *density_columns}
    for path in paths:
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if len(columns) != len(set(columns)):
                raise ValueError(f'{path} has duplicate column names')
            if not required.issubset(columns):
                raise ValueError(f'{path} lacks required columns {sorted(required - set(columns))}')
            for line, row in enumerate(reader, start=2):
                location = f'{path}, line {line}'
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'{location} has an invalid column count')
                cell_id = int(row['cell_id'])
                coordinates = [_finite_number(row[axis], location) for axis in ('x', 'y', 'z')]
                coordinate_key = _coordinate_key(coordinates)
                volume = _finite_number(row['volume'], location)
                if volume <= 0:
                    raise ValueError(f'{location} volume must be positive')
                geometry_text = row.get('cell_geometry_gid', '').strip()
                has_geometry = bool(geometry_text)
                geometry_columns.add(has_geometry)
                geometry_id = int(geometry_text) if has_geometry else None
                if cell_id in seen_ids or coordinate_key in seen_coordinates:
                    raise ValueError(f'{location} has a duplicate owned cell ID or coordinate')
                if has_geometry and geometry_id in seen_geometry_ids:
                    raise ValueError(f'{location} has duplicate cell_geometry_gid {geometry_id}')
                seen_ids.add(cell_id)
                seen_coordinates.add(coordinate_key)
                seen_geometry_ids.add(geometry_id)
                values = {}
                scales = {}
                for column in density_columns:
                    # Optional turbulence fields may be empty in legacy diagnostics.
                    values[column] = (_finite_number(row[column], location) if row[column] else None)
                    scale_column = column + '_algebraic_scale_density'
                    scales[column] = (_finite_number(row[scale_column], location)
                                      if scale_column in row and row[scale_column] else None)
                    if scales[column] is not None and scales[column] < 0:
                        raise ValueError(f'{location} has negative {scale_column}')
                cells.append(dict(cell_id=cell_id, cell_geometry_gid=geometry_id,
                                  coordinates=coordinates, coordinate_key=coordinate_key,
                                  volume=volume, values=values, scales=scales,
                                  integrated_residuals={column: row.get(column + '_integrated_residual')
                                                        for column in density_columns},
                                  integrated_scales={column: row.get(column + '_integrated_algebraic_scale')
                                                     for column in density_columns},
                                  accumulation_terms={column: row.get(column + '_accumulation_terms')
                                                      for column in density_columns},
                                  accumulation_eligible={column: row.get(column + '_accumulation_eligible')
                                                         for column in density_columns}))
    if len(geometry_columns) != 1:
        raise ValueError(f'{directory} mixes populated and unavailable cell_geometry_gid values')
    if not cells:
        raise ValueError(f'{directory} contains no owned cells')
    return cells, geometry_columns == {True}


def _matched_cells(serial_directory, parallel_directory):
    serial_cells, serial_geometry = _read_cells(serial_directory)
    parallel_cells, parallel_geometry = _read_cells(parallel_directory)
    use_geometry = serial_geometry and parallel_geometry
    key = 'cell_geometry_gid' if use_geometry else 'coordinate_key'
    serial = {cell[key]: cell for cell in serial_cells}
    parallel = {cell[key]: cell for cell in parallel_cells}
    if serial.keys() != parallel.keys():
        raise ValueError(f'Serial/parallel owned-cell sets differ when matched by {key}')
    pairs = []
    for identity in sorted(serial):
        a, b = serial[identity], parallel[identity]
        if a['coordinate_key'] != b['coordinate_key']:
            raise ValueError(f'Cell {identity} has mismatched serial/parallel coordinates')
        if not math.isclose(a['volume'], b['volume'], rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f'Cell {identity} has mismatched volumes '
                             f"serial={a['volume']!r}, parallel={b['volume']!r}")
        pairs.append((a, b))
    return pairs, use_geometry


def _cell_difference(pair, column, use_geometry):
    a, b = pair
    return dict(cell_geometry_gid=a['cell_geometry_gid'] if use_geometry else None,
                serial_cell_id=a['cell_id'], parallel_cell_id=b['cell_id'],
                coordinates=a['coordinates'], serial_coordinates=a['coordinates'],
                parallel_coordinates=b['coordinates'],
                serial_volume=a['volume'], parallel_volume=b['volume'],
                serial_value=a['values'][column], parallel_value=b['values'][column],
                absolute_difference=abs(a['values'][column] - b['values'][column]),
                serial_algebraic_scale_density=a['scales'][column],
                parallel_algebraic_scale_density=b['scales'][column])


def _argmax(pairs, side, column, use_geometry):
    maximum = max(abs(pair[side]['values'][column]) for pair in pairs)
    maxima = [pair for pair in pairs if abs(pair[side]['values'][column]) == maximum]
    cell = maxima[0][side]
    return dict(cell_geometry_gid=cell['cell_geometry_gid'] if use_geometry else None,
                cell_id=cell['cell_id'], coordinates=cell['coordinates'], volume=cell['volume'],
                value=cell['values'][column], algebraic_scale_density=cell['scales'][column],
                tie_count=len(maxima))


def _evaluation_precision(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Residual evaluation metadata unavailable')
    if metadata.get('kind') != 'crs_row_dot_minus_rhs':
        raise ValueError('Residual evaluation is not a CRS row dot product minus RHS')
    radix = metadata.get('scalar_radix')
    digits = metadata.get('scalar_digits')
    if type(radix) is not int or radix != 2:
        raise ValueError('Scalar radix must be binary')
    if type(digits) is not int or not 1 <= digits <= 53:
        raise ValueError('Scalar precision must be between 1 and 53 binary digits')
    raw_roundoff = metadata.get('unit_roundoff')
    if isinstance(raw_roundoff, bool) or not isinstance(raw_roundoff, (int, float)):
        raise ValueError('Scalar unit roundoff must be a number')
    roundoff = _finite_number(raw_roundoff, 'Scalar unit roundoff')
    if Fraction.from_float(roundoff) != Fraction(1, 2 ** digits):
        raise ValueError('Scalar unit roundoff does not match its binary precision')
    if metadata.get('round_to_nearest') is not True:
        raise ValueError('Round-to-nearest evaluation is unavailable')
    assumptions = metadata.get('arithmetic_assumptions')
    if not isinstance(assumptions, dict):
        raise ValueError('Arithmetic assumptions unavailable')
    if assumptions.get('iec559_binary') is not True:
        raise ValueError('IEC 559 binary arithmetic is unavailable')
    if assumptions.get('fast_math_disabled') is not True:
        raise ValueError('Reassociation and unsafe fast-math exclusion is unavailable')
    if assumptions.get('underflow_checks') != 'normal_or_zero_intermediates':
        raise ValueError('Normal-range intermediate evaluation checks are unavailable')
    return digits, Fraction.from_float(roundoff)


def _nonnegative_integer(value, description):
    if isinstance(value, bool):
        raise ValueError(f'{description} must be a nonnegative integer')
    if isinstance(value, str):
        text = value.strip()
        if not text or not text.isascii() or not text.isdecimal():
            raise ValueError(f'{description} must be a nonnegative integer')
        value = int(text)
    if type(value) is not int or value < 0:
        raise ValueError(f'{description} must be a nonnegative integer')
    return value


def integrated_row_comparison(serial_cell, parallel_cell, component,
                              serial_evaluation, parallel_evaluation):
    """Compare stored integrated rows under an explicit zero upstream allowance.

    Cells use _read_cells' dictionaries. The four integrated-data mappings are
    keyed by CSV component name and can hold raw CSV strings or numeric values.
    Classification uses exact rationals recovered from the parsed binary values.
    """
    result = dict(
        component=component, cell_geometry_gid=serial_cell.get('cell_geometry_gid'),
        serial_cell_id=serial_cell.get('cell_id'), parallel_cell_id=parallel_cell.get('cell_id'),
        serial_volume=serial_cell.get('volume'), parallel_volume=parallel_cell.get('volume'),
        status='unavailable', serial_integrated_residual=None, parallel_integrated_residual=None,
        serial_integrated_algebraic_scale=None, parallel_integrated_algebraic_scale=None,
        signed_difference=None, absolute_difference=None, unit_roundoff=None,
        serial_accumulation_terms=None, parallel_accumulation_terms=None,
        serial_operation_count=None, parallel_operation_count=None,
        accumulation_coefficient=None, tau=0.0, tau_status='upstream_unassessed',
        accumulation_bound=None, within_accumulation_envelope=None,
        required_tau_diagnostic=None, unavailable_reason=None)
    numeric_overflows = []

    def report_number(name, value):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if math.isfinite(number):
            result[name] = number
        else:
            numeric_overflows.append(name)

    try:
        if serial_cell.get('cell_geometry_gid') is None:
            raise ValueError('True cell_geometry_gid unavailable')
        if serial_cell.get('cell_geometry_gid') != parallel_cell.get('cell_geometry_gid'):
            raise ValueError('Serial/parallel cell_geometry_gid values differ')
        residuals = []
        scales = []
        for side, cell in (('serial', serial_cell), ('parallel', parallel_cell)):
            raw_residual = cell.get('integrated_residuals', {}).get(component)
            raw_scale = cell.get('integrated_scales', {}).get(component)
            if raw_residual is None or raw_residual == '' or raw_scale is None or raw_scale == '':
                raise ValueError(f'{side} stored integrated residual or algebraic scale unavailable')
            if isinstance(raw_residual, bool) or isinstance(raw_scale, bool):
                raise ValueError(f'{side} stored integrated values must be numbers')
            residual = _finite_number(raw_residual, f'{side} stored integrated residual')
            scale = _finite_number(raw_scale, f'{side} stored integrated algebraic scale')
            if scale < 0:
                raise ValueError(f'{side} integrated algebraic scale must be nonnegative')
            result[side + '_integrated_residual'] = residual
            result[side + '_integrated_algebraic_scale'] = scale
            residuals.append(Fraction.from_float(residual))
            scales.append(Fraction.from_float(scale))
        difference = residuals[0] - residuals[1]
        report_number('signed_difference', difference)
        report_number('absolute_difference', abs(difference))
        serial_precision = _evaluation_precision(serial_evaluation)
        parallel_precision = _evaluation_precision(parallel_evaluation)
        if serial_precision != parallel_precision:
            raise ValueError('Serial/parallel Scalar precision metadata differs')
        roundoff = serial_precision[1]
        report_number('unit_roundoff', roundoff)
        terms = []
        for side, cell in (('serial', serial_cell), ('parallel', parallel_cell)):
            count = _nonnegative_integer(cell.get('accumulation_terms', {}).get(component),
                                         f'{side} accumulation terms')
            result[side + '_accumulation_terms'] = count
            result[side + '_operation_count'] = 2 * count
            terms.append(count)
            eligible = cell.get('accumulation_eligible', {}).get(component)
            if not (eligible == '1' or eligible is True or (type(eligible) is int and eligible == 1)):
                raise ValueError(f'{side} row is excluded or lacks eligible normal-range, round-to-nearest evaluation')
        operation_counts = [2 * count for count in terms]
        if any(2 * count * roundoff >= 1 for count in operation_counts):
            raise ValueError('Accumulation model requires 2*k*u < 1 on each side')
        # k=2m covers m products and m additions, including separate rounding.
        # beta=gamma_k/(1-gamma_k) relates the error to the stored scale.
        betas = [count * roundoff / (1 - 2 * count * roundoff)
                 for count in operation_counts]
        coefficient = sum(betas) / roundoff
        allowance = sum(betas) * max(scales)
        required_tau = max(Fraction(0), abs(difference) - allowance)
        within = abs(difference) <= allowance
        result['status'] = 'within' if within else 'exceeds'
        result['within_accumulation_envelope'] = within
        report_number('accumulation_coefficient', coefficient)
        report_number('accumulation_bound', allowance)
        report_number('required_tau_diagnostic', required_tau)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        result['unavailable_reason'] = str(error)
    if numeric_overflows:
        result['numeric_range_note'] = ('Values outside the finite JSON number range are null. '
                                        'Classification uses exact rational arithmetic.')
        result['numeric_range_unavailable_fields'] = numeric_overflows
    return result


def _integrated_residual_check(serial_directory, parallel_directory, serial, parallel):
    check = dict(
        status='unavailable', purpose='Additional integrated residual diagnostic only',
        matching='cell_geometry_gid', units='integrated residual units for each component',
        model='abs(r_serial-r_parallel) <= tau + C*u*max(s_serial,s_parallel)',
        operation_model='k=2*m, beta=k*u/(1-2*k*u), C=(beta_serial+beta_parallel)/u',
        tau=0.0, tau_status='upstream_unassessed',
        tau_assumption='Zero upstream allowance is an explicit diagnostic assumption.',
        fixed_input_assumption='Identical assembled coefficients, RHS and fields are unverified. '
                               'Serial and MPI assembly may differ.',
        arithmetic_assumptions='Eligible rows require binary round-to-nearest arithmetic, '
                               'no reassociation or unsafe fast math, and normal-or-zero '
                               'evaluation intermediates.',
        required_tau_note='Observed excess above the accumulation envelope is a diagnostic. '
                          'It is not an adopted upstream tolerance.',
        counts=dict(checked=0, within=0, exceeds=0, unavailable=0), components={},
        unavailable_reason=None)
    try:
        pairs, use_geometry = _matched_cells(serial_directory, parallel_directory)
        if not use_geometry:
            raise ValueError('True cell_geometry_gid unavailable. Integrated rows require geometry global IDs.')
        check['matched_cell_count'] = len(pairs)
        for component in ('momentum_x', 'momentum_y', 'momentum_z', 'continuity', 'k', 'secondary'):
            if component.startswith('momentum_'):
                axis = ('momentum_x', 'momentum_y', 'momentum_z').index(component)
                serial_summary, parallel_summary = serial['momentum'][axis], parallel['momentum'][axis]
            else:
                summary_key = 'reconstructed_continuity' if component == 'continuity' else component
                serial_summary, parallel_summary = serial[summary_key], parallel[summary_key]
            cells = [integrated_row_comparison(
                a, b, component, serial_summary.get('residual_evaluation'),
                parallel_summary.get('residual_evaluation')) for a, b in pairs]
            counts = dict(checked=0, within=0, exceeds=0, unavailable=0)
            for cell in cells:
                counts[cell['status']] += 1
                if cell['status'] != 'unavailable':
                    counts['checked'] += 1
            check['components'][component] = dict(counts=counts, cells=cells)
            for name in counts:
                check['counts'][name] += counts[name]
        check['status'] = 'available' if check['counts']['unavailable'] == 0 else 'partial'
        if check['counts']['checked'] == 0:
            check['status'] = 'unavailable'
            check['unavailable_reason'] = 'No matched component rows have an available accumulation bound'
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        check['unavailable_reason'] = f'{type(error).__name__}: {error}'
    return check


def compare_replays(serial_directory, parallel_directory):
    """Return original summary acceptance and independent integrated row diagnostics."""
    serial = json.loads((Path(serial_directory) / 'diagnostics_step0.json').read_text())
    parallel = json.loads((Path(parallel_directory) / 'diagnostics_step0.json').read_text())
    comparisons = []
    for component in COMPONENTS:
        if component.startswith('momentum_'):
            axis = ('momentum_x', 'momentum_y', 'momentum_z').index(component)
            a, b = serial['momentum'][axis], parallel['momentum'][axis]
            unit_component = 'momentum'
        else:
            a, b = serial[component], parallel[component]
            unit_component = 'continuity' if component == 'reconstructed_continuity' else component
        for norm in NORMS:
            av, bv = a[norm], b[norm]
            comparisons.append(dict(
                component=component, norm=norm, serial_value=av, parallel_value=bv,
                absolute_difference=abs(av - bv),
                serial_algebraic_scale=a['scale'], parallel_algebraic_scale=b['scale'],
                algebraic_scale_kind='algebraic_absolute_row_terms_density_rms',
                serial_algebraic_scale_units=serial.get('residual_density_units', {}).get(unit_component),
                parallel_algebraic_scale_units=parallel.get('residual_density_units', {}).get(unit_component),
                allowed_difference=max(ABSOLUTE_TOLERANCE,
                                       RELATIVE_TOLERANCE * max(abs(av), abs(bv))),
                passed=math.isclose(av, bv, rel_tol=RELATIVE_TOLERANCE,
                                    abs_tol=ABSOLUTE_TOLERANCE)))
    failures = [comparison for comparison in comparisons if not comparison['passed']]
    report = dict(passed=not failures, relative_tolerance=RELATIVE_TOLERANCE,
                  absolute_tolerance=ABSOLUTE_TOLERANCE, comparisons=comparisons, failures=failures,
                  cell_evidence=dict(status='not_requested' if not failures else 'pending',
                                     purpose='Diagnostic localization only'))
    if failures:
        try:
            pairs, use_geometry = _matched_cells(serial_directory, parallel_directory)
            cell_evidence = dict(
                status='available', purpose='Diagnostic localization only',
                matching='cell_geometry_gid' if use_geometry else 'coordinate_key',
                geometry_id_available=use_geometry, matched_cell_count=len(pairs))
            if not use_geometry:
                cell_evidence['note'] = (
                    'Geometry global ID unavailable. Coordinate matching uses the replay 1e12 key. '
                    'Serial and parallel cell_id values are algebraic global IDs.')
            cell_details = []
            for failure in failures:
                column = ('continuity' if failure['component'] == 'reconstructed_continuity'
                          else failure['component'])
                if any(a['values'][column] is None or b['values'][column] is None for a, b in pairs):
                    raise ValueError(f'Cell residual density unavailable for {failure["component"]}')
                worst = max(pairs, key=lambda pair: abs(pair[0]['values'][column] - pair[1]['values'][column]))
                detail = dict(worst_cell_difference=_cell_difference(worst, column, use_geometry))
                if failure['norm'] == 'max_density':
                    detail['serial_argmax'] = _argmax(pairs, 0, column, use_geometry)
                    detail['parallel_argmax'] = _argmax(pairs, 1, column, use_geometry)
                    detail['argmax_note'] = (
                        'Each argmax identifies an aggregation maximum. The worst cell difference '
                        'is separate diagnostic evidence. tie_count includes all exact ties.')
                cell_details.append(detail)
            for failure, detail in zip(failures, cell_details):
                failure.update(detail)
            report['cell_evidence'] = cell_evidence
        except (OSError, ValueError, TypeError, KeyError) as error:
            report['evidence_error'] = f'{type(error).__name__}: {error}'
            report['cell_evidence']['status'] = 'unavailable'
    report['integrated_residual_check'] = _integrated_residual_check(
        serial_directory, parallel_directory, serial, parallel)
    return report


def format_comparison(report):
    """Format summary failures and the integrated residual diagnostic for stderr."""
    lines = [f"Replay comparison failed for {len(report['failures'])} of {len(report['comparisons'])} summaries.",
             'Algebraic scales below are global volume-weighted density RMS values.',
             'signed_sum uses integrated residual units. Cell differences are diagnostic localization only.']
    for failure in report['failures']:
        lines.append(
            f"{failure['component']} {failure['norm']} "
            f"serial={failure['serial_value']!r} parallel={failure['parallel_value']!r} "
            f"absolute_difference={failure['absolute_difference']!r} "
            f"allowed_difference={failure['allowed_difference']!r} "
            f"serial_algebraic_scale={failure['serial_algebraic_scale']!r} "
            f"parallel_algebraic_scale={failure['parallel_algebraic_scale']!r}")
        if 'worst_cell_difference' in failure:
            cell = failure['worst_cell_difference']
            lines.append('  worst_cell_difference ' + json.dumps(cell, sort_keys=True))
            if (cell['serial_algebraic_scale_density'] is None
                    or cell['parallel_algebraic_scale_density'] is None):
                lines.append('  Cell algebraic scales unavailable for one or both replays in retained CSV evidence.')
        if 'serial_argmax' in failure:
            lines.append('  serial_argmax ' + json.dumps(failure['serial_argmax'], sort_keys=True))
            lines.append('  parallel_argmax ' + json.dumps(failure['parallel_argmax'], sort_keys=True))
            lines.append('  ' + failure['argmax_note'])
    if 'note' in report['cell_evidence']:
        lines.append(report['cell_evidence']['note'])
    if 'evidence_error' in report:
        lines.append('Cell evidence unavailable. ' + report['evidence_error'])
    integrated = report.get('integrated_residual_check')
    if integrated is not None:
        lines.append('Integrated residual diagnostic ' + integrated['status'] + ' ' +
                     json.dumps(integrated['counts'], sort_keys=True))
        lines.append('  Integrated residuals and algebraic scales use integrated component units. '
                     'tau=0 is assumed, tau_status=upstream_unassessed. '
                     'Identical assembled coefficients, RHS and fields are unverified.')
        if integrated['unavailable_reason']:
            lines.append('  ' + integrated['unavailable_reason'])
        for component, detail in integrated['components'].items():
            lines.append('  ' + component + ' ' + json.dumps(detail['counts'], sort_keys=True))
            exceeded = [cell for cell in detail['cells'] if cell['status'] == 'exceeds']
            if exceeded:
                worst = max(exceeded, key=lambda cell: (
                    cell['required_tau_diagnostic'] if cell['required_tau_diagnostic'] is not None
                    else math.inf))
                lines.append('    largest_required_tau_diagnostic ' + json.dumps(worst, sort_keys=True))
            unavailable = next((cell for cell in detail['cells']
                                if cell['status'] == 'unavailable'), None)
            if unavailable is not None:
                lines.append('    bound unavailable. ' + unavailable['unavailable_reason'])
    return '\n'.join(lines)
