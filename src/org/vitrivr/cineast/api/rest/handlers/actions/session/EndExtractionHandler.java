package org.vitrivr.cineast.api.rest.handlers.actions.session;

import java.util.Map;
import org.vitrivr.cineast.api.SessionExtractionContainer;
import org.vitrivr.cineast.api.rest.exceptions.ActionHandlerException;
import org.vitrivr.cineast.api.rest.handlers.abstracts.ParsingActionHandler;
import org.vitrivr.cineast.core.data.messages.general.AnyMessage;
import org.vitrivr.cineast.core.data.messages.session.SessionState;

/**
 * @author silvan on 23.01.18.
 */
public class EndExtractionHandler extends ParsingActionHandler<AnyMessage> {

  @Override
  public Object doGet(Map<String, String> parameters) throws ActionHandlerException {
    throw new UnsupportedOperationException();
  }

  @Override
  public Object doPost(AnyMessage context, Map<String, String> parameters)
      throws ActionHandlerException {
    SessionState state = ValidateSessionHandler.validateSession(parameters); //TODO Use State

    SessionExtractionContainer.endSession();
    return state;
  }

  @Override
  public Class<AnyMessage> inClass() {
    return AnyMessage.class;
  }
}
