function stop_recording(operator_id)
    arguments
        operator_id (1,1) uint16 = 0
    end

    recording_stopped_automatically = 0;

    try
        xippmex('trial', 'paused', [], [], [], [], double(operator_id));
        trial_descriptor = xippmex('trial', 'stopped', [], [], [], [], double(operator_id));
        if strcmp(trial_descriptor.status, 'stopped')
            recording_stopped_automatically = 1;
            fprintf('sEEG recording stopped automatically!\n');
        end
    catch
    end

    if ~recording_stopped_automatically
        warning('sEEG recording may not have been stopped automatically. Stop the recording manually in the Trellis GUI now.\n');

        recording_stopped_manually = input('Confirm that the recording has been stopped manually and that the output files have been saved [1 for yes, 0 for no]: ');
        if recording_stopped_manually
            fprintf('sEEG recording stopped manually!\n');
        else
            error('sEEG recording was not stopped automatically or manually.\n')
        end
    end
end
