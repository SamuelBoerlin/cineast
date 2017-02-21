package org.vitrivr.cineast.core.data.messages.lookup;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import org.vitrivr.cineast.core.data.messages.interfaces.Message;
import org.vitrivr.cineast.core.data.messages.interfaces.MessageTypes;

/**
 * @author rgasser
 * @version 1.0
 * @created 10.02.17
 */
public class MetadataLookup implements Message {
    /**
     *
     */
    private String[] objectids;

    /**
     *
     */
    private String[] domains;

    /**
     *
     * @param objectids
     * @param domains
     */
    @JsonCreator
    public MetadataLookup(@JsonProperty("objectids") String[] objectids, @JsonProperty("domains") String[] domains) {
        this.objectids = objectids;
        this.domains = domains;
    }

    /**
     *
     * @return
     */
    public String[] getObjectids() {
        return this.objectids;
    }

    /**
     *
     * @return
     */
    public String[] getDomains() {
        return this.domains;
    }

    /**
     * Returns the type of particular message. Expressed as MessageTypes enum.
     *
     * @return
     */
    @Override
    public MessageTypes getMessagetype() {
        return null;
    }
}