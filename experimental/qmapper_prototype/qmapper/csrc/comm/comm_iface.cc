#include "comm_iface.h"
#include <ucs/type/status.h>

comm_status_t qmap::comm::ucs_status_to_qmap_status(ucs_status_t status) {
    switch(status) {
        case UCS_OK:
            return COMM_OK;
        case UCS_INPROGRESS:
            return COMM_INPROGRESS;
        default:
            break;
    }
    return COMM_ERROR;
}

ucs_status_t qmap::comm::qmap_status_to_ucs_status(comm_status_t status) {
    switch(status) {
        case COMM_OK:
            return UCS_OK;
        case COMM_INPROGRESS:
            return UCS_INPROGRESS;
        default:
            break;
    }
    return UCS_ERR_NO_MESSAGE;
}