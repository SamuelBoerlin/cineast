package org.vitrivr.cineast.api.rest.handlers.actions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.vitrivr.cineast.api.rest.exceptions.ActionHandlerException;
import org.vitrivr.cineast.api.rest.handlers.abstracts.ParsingActionHandler;
import org.vitrivr.cineast.core.data.messages.lookup.IdList;
import org.vitrivr.cineast.core.data.tag.Tag;
import org.vitrivr.cineast.core.db.dao.TagHandler;

public class FindTagsByActionHandler extends ParsingActionHandler<IdList> {
    /** The {@link TagHandler} instance used for lookup of {@link Tag}s. */
    private static final TagHandler TAG_HANDLER = new TagHandler();

    /** */
    private static final Logger LOGGER = LogManager.getLogger();

    /** Name of the GET parameter that denotes the attribute that should be used for lookup. */
    private static final String GET_PARAMETER_ATTRIBUTE = ":attribute";

    /** Name of the GET parameter that denotes the value that should be used for lookup. */
    private static final String GET_PARAMETER_VALUE = ":value";

    /** Supported field names (attributes). */
    private static final String FIELD_ID = "id";
    private static final String FIELD_NAME = "name";
    private static final String FIELD_MATCHING = "matchingname";

    /**
     * Performs the lookup of {@link Tag}s in the system.
     *
     * TODO: Check if IdList etc. is still required.
     *
     * @param parameters Map containing named parameters in the URL.
     * @return
     * @throws ActionHandlerException
     */
    @Override
    public List<Tag> doGet(Map<String, String> parameters) {
        final String attribute = parameters.get(GET_PARAMETER_ATTRIBUTE);
        final String value = parameters.get(GET_PARAMETER_VALUE);
        switch (attribute.toLowerCase()) {
            case FIELD_ID:
                final List<Tag> list = new ArrayList<>(1);
                list.add(TAG_HANDLER.getTagById(value));
                return list;
            case FIELD_NAME:
                return TAG_HANDLER.getTagsByName(value);
            case FIELD_MATCHING:
                return TAG_HANDLER.getTagsByMatchingName(value);
            default:
                LOGGER.error("Unknown attribute '{}' in FindTagsByActionHandler", attribute);
                return new ArrayList<>(0);
        }
    }

    /**
     *
     * @param context Object that is handed to the invocation, usually parsed from the request body. May be NULL!
     * @param parameters Map containing named parameters in the URL.
     * @return
     */
    public List<Tag> doPost(IdList context, Map<String, String> parameters) {
        if (context == null || context.getIds().length == 0) {
            return Collections.emptyList();
        }
        return TAG_HANDLER.getTagsById(context.getIds());
    }


    @Override
    public Class<IdList> inClass() {
        return IdList.class;
    }
}
