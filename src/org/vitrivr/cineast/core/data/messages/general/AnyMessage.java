package org.vitrivr.cineast.core.data.messages.general;

import org.vitrivr.cineast.core.data.messages.interfaces.Message;
import org.vitrivr.cineast.core.data.messages.interfaces.MessageType;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * @author rgasser
 * @version 1.0
 * @created 19.01.17
 */
public class AnyMessage implements Message {
    private MessageType messageType;

    @Override
    @JsonProperty
    public MessageType getMessageType() {
        return this.messageType;
    }
    public void setMessagetype(MessageType messageType) {
        this.messageType = messageType;
    }
}
