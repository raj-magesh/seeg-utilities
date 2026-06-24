function send_trigger(trigger_value)
    arguments
        trigger_value (1,1) uint16 = 0
    end

    try
        xippmex('digout', 5, double(trigger_value));
        fprintf('Sent trigger %s to parallel port\n', num2str(trigger_value));
    catch
        warning('Failed to send trigger %s to parallel port\n', num2str(trigger_value));
    end
end
